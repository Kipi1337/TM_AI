# Phase 1 – Trackmania AI (Fast Success)

## Goal
Achieve fast lap times using input replay and hill-climbing, without reinforcement learning.

## Core Idea
Trackmania is deterministic. If an input sequence produces a good lap, slightly modifying it can produce a better one.

## What the AI Reads
- Real Speed (numeric only)
- Checkpoints (numeric only)

No labels, no position, no rewards.

## Race Logic
- Race starts when speed > 0
- Race ends at checkpoint 3/3
- Best lap time is stored
- Input sequence is saved
- Next run mutates timing slightly

## Why This Works
- Inputs matter more than perception
- OCR is only used for timing gates
- Hill-climbing converges quickly

## What This Is NOT
- Not reinforcement learning
- Not pathfinding
- Not physics-based steering

This phase proves the concept and should already approach competitive times.
