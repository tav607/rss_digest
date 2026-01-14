#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import logging
import re
from typing import Dict, Any

# 创建命名记录器
logger = logging.getLogger(__name__)

def _convert_markdown_bold_to_telegram(text: str) -> tuple[str, list[str]]:
    """
    Convert standard Markdown **bold** to Telegram MarkdownV2 bold markers.
    Returns the text with bold content replaced by placeholders and the bold segments.
    """
    bold_segments = []

    def replace_bold(match):
        content = match.group(1)
        bold_segments.append(content)
        # Use a placeholder that won't be escaped
        return f"\x00BOLD{len(bold_segments)-1}\x00"

    # Match **text** (standard Markdown bold)
    result = re.sub(r'\*\*([^*]+)\*\*', replace_bold, text)
    return result, bold_segments


def _convert_markdown_links_to_telegram(text: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Extract Markdown links [text](url) and replace with placeholders.
    Returns text with placeholders and list of (display_text, url) tuples.
    """
    links = []

    def replace_link(match):
        display_text = match.group(1)
        url = match.group(2)
        links.append((display_text, url))
        return f"\x00LINK{len(links)-1}\x00"

    # Match [text](url) - handle one level of nested parentheses in URLs (e.g. Wikipedia)
    # Pattern: [^()]* matches non-parens, (?:\([^()]*\)[^()]*)* matches balanced (...)
    result = re.sub(r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)', replace_link, text)
    return result, links


def _escape_markdown_v2_content(text: str) -> str:
    """Escapes MarkdownV2 special characters in a given string, preserving **bold** and [text](url) links."""
    # First, extract and protect links
    text_with_link_placeholders, links = _convert_markdown_links_to_telegram(text)

    # Then, extract and protect bold segments
    text_with_placeholders, bold_segments = _convert_markdown_bold_to_telegram(text_with_link_placeholders)

    # Characters to escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
    escape_chars_pattern = r'([_*\[\]()~`>#+=|{}.!-])'
    # Escape the characters by adding a preceding \
    escaped_text = re.sub(escape_chars_pattern, r'\\\1', text_with_placeholders)

    # Restore bold segments with proper Telegram MarkdownV2 formatting
    for i, content in enumerate(bold_segments):
        # Escape the content inside bold markers
        escaped_content = re.sub(escape_chars_pattern, r'\\\1', content)
        # Replace placeholder with Telegram bold format: *text*
        escaped_text = escaped_text.replace(f"\x00BOLD{i}\x00", f"*{escaped_content}*")

    # Restore links with proper MarkdownV2 format
    for i, (display_text, url) in enumerate(links):
        # Escape display text
        escaped_display = re.sub(escape_chars_pattern, r'\\\1', display_text)
        # Escape special chars in URL (only ) and \)
        escaped_url = url.replace('\\', '\\\\').replace(')', '\\)')
        escaped_text = escaped_text.replace(f"\x00LINK{i}\x00", f"[{escaped_display}]({escaped_url})")

    return escaped_text

# Note: Bold preservation is not handled specially; text is escaped uniformly.

def _process_markdown_structure_and_escape(text: str) -> str:
    """Processes specific markdown structure (#, ##, -) and escapes the rest for MarkdownV2."""
    processed_lines = []
    for line in text.strip().split('\n'):
        stripped_line = line.strip()
        if stripped_line.startswith('# '): # Main title
            # Escape content after '# '
            title_content = _escape_markdown_v2_content(stripped_line[2:])
            # Format as Bold
            processed_lines.append(f"*{title_content}*")
        elif stripped_line.startswith('## '): # Subtitle
            # Escape content after '## '
            subtitle_content = _escape_markdown_v2_content(stripped_line[3:])
            # Format as Bold
            processed_lines.append(f"*{subtitle_content}*")
        elif stripped_line.startswith('- '): # Bullet point
            # Escape content after '- '
            bullet_content = _escape_markdown_v2_content(stripped_line[2:])
            # Keep the bullet point structure, Telegram supports this
            processed_lines.append(f"\\- {bullet_content}")
        elif stripped_line: # Non-empty line, escape the whole line
            processed_lines.append(_escape_markdown_v2_content(stripped_line))
        else: # Keep empty lines for separation
            processed_lines.append("")
    return '\n'.join(processed_lines)

class TelegramSender:
    """
    Class for sending messages via Telegram bot
    """
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize the Telegram sender
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        
    def _send_single_message(self, text: str, parse_mode: str) -> Dict[str, Any]:
        """Internal method to send a single message chunk."""
        endpoint = f"{self.api_url}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": text
        }
        if parse_mode:
            data["parse_mode"] = parse_mode

        try:
            response = requests.post(endpoint, data=data)
            result = response.json()
            if not result.get("ok"):
                error_msg = f"Failed to send Telegram message chunk: {result.get('description', 'Unknown error')} for text starting with: {text[:50]}..."
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            return {"success": True, "result": result}
        except Exception as e:
            error_msg = f"Error sending Telegram message chunk: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def send_message(self, text: str, parse_mode: str = "MarkdownV2", process_markdown: bool = True) -> Dict[str, Any]:
        """
        Send a message via Telegram, processing custom markdown and handling long messages.

        Args:
            text: Text message to send.
            parse_mode: Message parse mode (MarkdownV2, HTML, or empty string). Defaults to MarkdownV2.
            process_markdown: If True (default), process text with #, ##, - and escape for MarkdownV2.

        Returns:
            Response from Telegram API for the last message part sent.
        """
        logger.info(f"Preparing message for Telegram, original length: {len(text)} characters")

        processed_text = text
        if parse_mode == "MarkdownV2":
            if process_markdown:
                logger.debug("Processing text for MarkdownV2 structure and escaping...")
                processed_text = _process_markdown_structure_and_escape(text)
            else:
                logger.debug("Escaping text content for MarkdownV2...")
                processed_text = _escape_markdown_v2_content(text)
            logger.debug(f"Processed text length: {len(processed_text)}")

        # Check length after processing/escaping
        if len(processed_text) > 4096:  # Telegram's hard limit
            logger.info(f"Message length ({len(processed_text)}) exceeds limit, splitting...")
            return self._send_long_message(processed_text, parse_mode)
        else:
            logger.info(f"Sending single message part, length: {len(processed_text)}")
            return self._send_single_message(processed_text, parse_mode)

    def _send_long_message(self, processed_text: str, parse_mode: str) -> Dict[str, Any]:
        """
        Send a long message (already processed/escaped) by splitting it carefully.

        Args:
            processed_text: The full text message, already processed and escaped.
            parse_mode: Message parse mode.

        Returns:
            Response from the last message sent.
        """
        # Split primarily by double newline, then single newline, then space
        chunks = []
        # Split carefully to avoid breaking mid-escape sequence or formatting
        potential_splits = re.split(r'(\n\n|\n)', processed_text)

        temp_chunk = ""
        for part in potential_splits:
            if part is None:
                continue  # Skip None parts if regex captures optional groups

            # If adding the next part (e.g., "\n\n" or text segment) exceeds the limit
            if len(temp_chunk) + len(part) <= 4096:
                temp_chunk += part
            else:
                # If the current temp_chunk is not empty, add it as a chunk
                if temp_chunk:
                    chunks.append(temp_chunk)
                # If the part itself is too long, we have to split it (less ideal)
                if len(part) > 4096:
                     # Force split the oversized part
                    for i in range(0, len(part), 4096):
                        chunks.append(part[i:i+4096])
                    temp_chunk = ""  # Reset temp_chunk
                else:
                    # Start the new chunk with this part
                    temp_chunk = part

        # Add the last remaining chunk
        if temp_chunk:
            chunks.append(temp_chunk)

        last_response = None
        total_chunks = len(chunks)
        logger.info(f"Splitting message into {total_chunks} chunks.")

        for i, chunk in enumerate(chunks):
            if not chunk.strip():  # Skip empty chunks
                continue
            logger.info(f"Sending chunk {i+1}/{total_chunks}, length: {len(chunk)}")
            response = self._send_single_message(chunk, parse_mode)
            last_response = response
            if not response.get("success"):
                logger.error(f"Failed to send chunk {i+1}, stopping further sends.")
                # Return the error from the failed chunk
                return response

        return last_response or {"success": False, "error": "No content chunks were sent"}
