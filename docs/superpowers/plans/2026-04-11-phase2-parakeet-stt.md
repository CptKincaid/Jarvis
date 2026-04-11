# Phase 2: Parakeet STT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Whisper small with NVIDIA Parakeet-TDT-0.6B-v2 for 7x faster, 3.4x more accurate speech-to-text.

**Architecture:** New `stt_engine.py` module wraps Parakeet with a clean interface. GUI swaps from direct faster-whisper calls to the new engine. Whisper kept as automatic fallback if Parakeet fails to load.

**Tech Stack:** nemo_toolkit (already installed), soundfile, numpy, faster-whisper (fallback)

---

## 6 Tasks — see full plan content in previous context.

Plan saved. Executing inline.
