from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from copy import deepcopy
from PIL import Image
from ddpo.ddpo_agent import DDPOAgent
from morl_utils.pgmorl import PerformanceBuffer, PerformancePredictor, ParetoArchive, generate_weights, hypervolume, sparsity

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # setup pgmorl
    ref_point = np.array([0.0, 0.0])
    population = PerformanceBuffer(
        num_bins=config.num_performance_buffer,
        max_size=config.performance_buffer_size,
        origin=ref_point,
    )
    predictor = PerformancePredictor()
    pareto_archive = ParetoArchive()
    weights = generate_weights(config.delta_weight)

    # setup the models
    agents = [DDPOAgent(id=i, config=config, weights=weights[i], accelerator=accelerator) for i in range(config.pop_size)]

    # prepare prompt and reward fn
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fns = [getattr(ddpo_pytorch.rewards, fn)() for fn in config.reward_fns]

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Total Epochs = {config.num_epochs}")
    logger.info(f"  Warm up epochs = {config.warmup_iterations}")
    logger.info(f"  Evolutionary iterations = {config.evolutionary_iterations}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    logger.info("")
    logger.info(f"  Warmup phase - sampled weights: {weights}")
    logger.info(f"  Population size: {config.pop_size}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    # Init
    current_evaluations = [np.zeros(config.num_reward_fns) for _ in range(config.pop_size)]
    evolutionary_generation = 1

    for epoch in range(first_epoch, config.num_epochs):
        if epoch < config.warmup_iterations:
            logger.info(f"Running warmup iteration {epoch}")
            for agent in agents:
                logger.info(f"Traning for agent {agent.id}")
                reward = agent.sampling(prompt_fn, reward_fns, executor, epoch)
                
                if epoch == config.warmup_iterations - 1:
                    # Storing results for last warmup iteration
                    population.add(agent, reward)
                    pareto_archive.add(agent, reward)
                    predictor.add(agent.weights, current_evaluations[agent.id], reward)
                    current_evaluations[agent.id] = reward
                
                agent.training(epoch)
        else:
            logger.info(f"Running evolutionary iteration {epoch - config.warmup_iterations}")
            # Every evolutionary iterations, change the task - weight assignments
            # Chooses agents and weights to train at the next iteration based on the current population and prediction model.
            candidate_weights = generate_weights(config.delta_weight / 2.0)  # Generates more weights than agents
            np.random.shuffle(candidate_weights)

            current_front = deepcopy(pareto_archive.evaluations)
            population_inds = population.individuals
            population_eval = population.evaluations
            selected_tasks = []
            # For each worker, select a (policy, weight) tuple
            for i in range(len(agents)):
                max_improv = float("-inf")
                best_candidate = None
                best_eval = None
                best_predicted_eval = None

                # In each selection, look at every possible candidate in the current population and every possible weight generated
                for candidate, last_candidate_eval in zip(population_inds, population_eval):
                    # Pruning the already selected (candidate, weight) pairs
                    candidate_tuples = [
                        (last_candidate_eval, weight)
                        for weight in candidate_weights
                        if (tuple(last_candidate_eval), tuple(weight)) not in selected_tasks
                    ]

                    # Prediction of improvements of each pair
                    delta_predictions, predicted_evals = map(
                        list,
                        zip(
                            *[
                                predictor.predict_next_evaluation(weight, candidate_eval)
                                for candidate_eval, weight in candidate_tuples
                            ]
                        ),
                    )
                    # optimization criterion is a hypervolume - sparsity
                    mixture_metrics = [
                        hypervolume(ref_point, current_front + [predicted_eval]) - sparsity(current_front + [predicted_eval])
                        for predicted_eval in predicted_evals
                    ]
                    # Best among all the weights for the current candidate
                    current_candidate_weight = np.argmax(np.array(mixture_metrics))
                    current_candidate_improv = np.max(np.array(mixture_metrics))

                    # Best among all candidates, weight tuple update
                    if max_improv < current_candidate_improv:
                        max_improv = current_candidate_improv
                        best_candidate = (
                            candidate,
                            candidate_tuples[current_candidate_weight][1],
                        )
                        best_eval = last_candidate_eval
                        best_predicted_eval = predicted_evals[current_candidate_weight]

                selected_tasks.append((tuple(best_eval), tuple(best_candidate[1])))
                # Append current estimate to the estimated front (to compute the next predictions)
                current_front.append(best_predicted_eval)

                # Assigns best predicted (weight-agent) pair to the worker
                copied_agent = deepcopy(best_candidate[0])
                copied_agent.global_step = agents[i].global_step
                copied_agent.id = i
                copied_agent.weights = best_candidate[1]
                agents[i] = copied_agent

                print(f"Agent #{agents[i].id} - weights {best_candidate[1]}")
                print(f"current eval: {best_eval} - estimated next: {best_predicted_eval} - deltas {(best_predicted_eval - best_eval)}")

            print(f"Evolutionary generation #{evolutionary_generation}")
            for _ in range(config.evolutionary_iterations):
                for agent in agents:
                    logger.info(f"Traning for agent {agent.id}")
                    reward = agent.sampling(prompt_fn, reward_fns, executor, epoch)
                    population.add(agent, reward)
                    pareto_archive.add(agent, reward)
                    predictor.add(agent.weights, current_evaluations[agent.id], reward)
                    current_evaluations[agent.id] = reward
                    agent.training(epoch)
            evolutionary_generation += 1

            # Save checkpoint
            if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
                accelerator.save_state()
        

if __name__ == "__main__":
    app.run(main)