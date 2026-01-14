#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import datetime
from typing import Dict, List, Any, Union
from openai import OpenAI
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 从配置导入API调试日志路径
from src.config import API_DEBUG_LOG_PATH
# No extra config imports required here; stage1/2 behavior is orchestrated in services.

# 创建命名记录器
logger = logging.getLogger(__name__)

# 为API调用创建专用的调试记录器
api_logger = logging.getLogger("api")
api_handler = logging.FileHandler(API_DEBUG_LOG_PATH)
api_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
api_logger.addHandler(api_handler)
api_logger.setLevel(logging.DEBUG)

# 读取 system prompt 内容（第二阶段：全局汇总）
SYSTEM_PROMPT = "你是一位资深新闻编辑，擅长从大量资讯中提取核心信息并生成摘要。"  # Default fallback
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "system_prompt.md")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read()
        logger.info(f"Successfully loaded system prompt from {prompt_path}")
    else:
        logger.error(f"System prompt file '{prompt_path}' not found. Using default prompt.")
except Exception as e:
    logger.error(f"Error reading system prompt file '{prompt_path}': {e}. Using default prompt.")

# 读取第一阶段（单篇文章摘要）提示词
STAGE1_SYSTEM_PROMPT = (
    "你是资深新闻编辑。基于输入的一篇完整文章（含标题、来源、正文），提炼要点列表。\n\n"
    "规则：\n"
    "1. 生成 1-3 条要点，按重要度排序；信息不足可少于此数，但不要杜撰。\n"
    "2. 每条采用「**主体** + 动作 + 目的/影响」句式，≤ 40 字。\n"
    "3. 主体必须是明确的公司/机构/产品名词，英文公司名保持英文（如 Apple、Nvidia、OpenAI）。\n"
    "4. 只保留增量信息，避免背景铺陈与主观评价。\n"
    "5. 末尾标注分类建议：[AI] [Semi] [Smartphone] [Other Tech] [World News] [Misc]（可多选）\n\n"
    "输出格式：\n"
    "- **主体** 动作内容（≤40字）\n"
    "- ...\n"
    "[分类: AI, Smartphone]"
)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path_stage1 = os.path.join(script_dir, "system_prompt_stage1.md")
    if os.path.exists(prompt_path_stage1):
        with open(prompt_path_stage1, "r", encoding="utf-8") as f:
            STAGE1_SYSTEM_PROMPT = f.read()
        logger.info(f"Successfully loaded stage1 system prompt from {prompt_path_stage1}")
    else:
        logger.info(
            f"Stage1 system prompt file '{prompt_path_stage1}' not found. Using default stage1 prompt."
        )
except Exception as e:
    logger.error(
        f"Error reading stage1 system prompt file '{prompt_path_stage1}': {e}. Using default stage1 prompt."
    )


class AIProcessor:
    """
    通过 AI 模型处理内容并生成摘要的类。
    """
    def __init__(self, api_key: str, stage2_model: str, base_url: str, stage1_model: str = None):
        """
        初始化 AI 处理器。

        Args:
            api_key: AI 服务的 API 密钥。
            stage2_model: 第二阶段（全局汇总）默认使用的模型。
            base_url: API 的基础 URL。
        """
        self.stage2_model = stage2_model
        self.stage1_model = stage1_model or stage2_model
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        api_logger.debug(
            "AIProcessor initialized with stage1_model=%s, stage2_model=%s, base_url=%s",
            self.stage1_model,
            self.stage2_model,
            base_url,
        )

    def _truncate_content(self, content: str, max_chars: int = 4000) -> str:
        """将内容截断到合理长度"""
        if content is None:
            return ""
        content_str = str(content)
        if len(content_str) <= max_chars:
            return content_str
        return content_str[:max_chars] + "..."

    def _get_formatted_date(self) -> str:
        """获取中文格式的当前日期"""
        now = datetime.datetime.now()
        return f"{now.year}年{now.month}月{now.day}日"

    # -------------------------
    # 第一阶段：逐篇文章摘要
    # -------------------------
    def summarize_single_article(self, entry: Dict[str, Any]) -> str:
        """
        对单篇 RSS 文章进行摘要，返回若干条一行要点。
        不进行内容截断（遵从用户对预算不敏感的偏好）。
        """
        title = entry.get('title', 'N/A')
        source = entry.get('feed_name', 'N/A')
        content = entry.get('content', '') or ''

        user_prompt = (
            "请基于以下单篇文章内容提炼要点，遵循系统提示的格式要求：\n\n"
            f"标题: {title}\n来源: {source}\n正文:\n{content}\n"
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.stage1_model,
                messages=[
                    {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            summary = (completion.choices[0].message.content or "").strip()
            api_logger.debug(
                f"Stage1 article summary generated: title='{title[:60]}', len={len(summary)}"
            )
            return summary or ""
        except Exception as e:
            api_logger.error(f"Stage1 article summary error for '{title[:60]}': {e}")
            return ""

    def summarize_articles(self, entries: List[Dict[str, Any]]) -> str:
        """
        逐篇调用 summarize_single_article，合并为一个文本，供第二阶段汇总。
        输出格式：每篇以“--- ARTICLE ---”分隔，包含来源与要点。
        """
        if not entries:
            return ""
        from src.config import STAGE1_MAX_WORKERS  # import here to avoid early import side-effects

        def worker(i: int, e: Dict[str, Any]):
            # Create a lightweight client per thread for safety
            local_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            title = e.get('title', 'N/A')
            source = e.get('feed_name', 'N/A')
            link = e.get('link', '')
            content = e.get('content', '') or ''
            user_prompt = (
                "请基于以下单篇文章内容提炼要点，遵循系统提示的格式要求：\n\n"
                f"标题: {title}\n来源: {source}\n原文链接: {link}\n正文:\n{content}\n"
            )
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    completion = local_client.chat.completions.create(
                        model=self.stage1_model,
                        messages=[
                            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                    )
                    choice = completion.choices[0] if completion and completion.choices else None
                    summary = (choice.message.content if choice and choice.message else "")
                    summary = (summary or "").strip()
                    if summary:
                        if attempt > 1:
                            api_logger.debug(
                                f"Stage1 success on retry {attempt} for title='{title[:60]}'"
                            )
                        return i, title, source, link, summary
                    else:
                        raise RuntimeError("empty summary")
                except Exception as ex:
                    api_logger.warning(
                        f"Stage1 attempt {attempt}/{max_attempts} failed for '{title[:60]}': {ex}"
                    )
                    if attempt < max_attempts:
                        backoff = 0.3 * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                        time.sleep(backoff)
                        continue
                    else:
                        api_logger.error(
                            f"Stage1 giving up for '{title[:60]}' after {max_attempts} attempts"
                        )
                        return i, title, source, link, ""

        api_logger.debug(
            f"Stage1 parallel summarization start: entries={len(entries)}, max_workers={STAGE1_MAX_WORKERS}"
        )
        results: List[Any] = [None] * len(entries)
        with ThreadPoolExecutor(max_workers=STAGE1_MAX_WORKERS) as executor:
            futures = [executor.submit(worker, idx, entry) for idx, entry in enumerate(entries, start=1)]
            for fut in as_completed(futures):
                i, title, source, link, per_article = fut.result()
                header = f"[来源:{source}] [链接:{link}] [标题:{title}]"
                block = (
                    f"--- ARTICLE {i} START ---\n{header}\n要点:\n{per_article}\n--- ARTICLE {i} END ---"
                )
                results[i - 1] = block

        parts: List[str] = [blk for blk in results if blk is not None]
        merged = "\n\n".join(parts)
        api_logger.debug(
            f"Stage1 merged summaries generated for {len(entries)} articles, length={len(merged)}"
        )
        return merged

    # -------------------------
    # 第二阶段：基于文章摘要进行全局汇总
    # -------------------------
    def finalize_digest_from_article_summaries(
        self, merged_summaries: str, digest_history: list[str] = None
    ) -> str:
        """
        采用现有的 system prompt，对第一阶段的合并摘要做全局分类与排序，生成终稿。

        Args:
            merged_summaries: 第一阶段合并的文章摘要
            digest_history: 最近几次的历史摘要，用于去重
        """
        if not merged_summaries or not merged_summaries.strip():
            return ""

        # Build user prompt with optional history context
        user_content_parts = []

        # Add history context if available
        if digest_history:
            history_text = "\n\n---\n\n".join(digest_history)
            user_content_parts.append(
                "以下是最近已发送的摘要（用于去重参考）：\n"
                "--- HISTORY_START ---\n"
                f"{history_text}\n"
                "--- HISTORY_END ---\n"
            )

        # Add current batch
        user_content_parts.append(
            "以下为本次待处理的文章要点摘要：\n"
            "--- ABSTRACT_BATCH_START ---\n\n"
            f"{merged_summaries}\n\n"
            "--- ABSTRACT_BATCH_END ---"
        )

        user_content = "\n".join(user_content_parts)

        try:
            completion = self.client.chat.completions.create(
                model=self.stage2_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=1.0,
            )
            digest = (completion.choices[0].message.content or "").strip()
            api_logger.debug(
                f"Stage2 digest generated from abstracts, length={len(digest)}"
            )
            return digest
        except Exception as e:
            api_logger.error(f"Stage2 digest generation error: {e}")
            return ""
