"""
Chat Audit Logger
Comprehensive logging for all chat interactions with DeepSeek AI.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


class ChatAuditLogger:
    """
    Comprehensive audit logger for chat interactions.
    Logs all user messages, AI responses, and system actions.
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory to store audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"chat_audit_{timestamp}.log"

        # Session tracking
        self.session_id = None
        self.session_start = None

        logger.info(f"Chat audit logger initialized: {self.log_file}")

    def start_session(self) -> str:
        """
        Start a new chat session.

        Returns:
            Session ID
        """
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_start = datetime.now()

        self.log_event({
            "event_type": "SESSION_START",
            "session_id": self.session_id,
            "timestamp": self.session_start.isoformat()
        })

        return self.session_id

    def log_user_message(
        self,
        message: str,
        message_type: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log user message.

        Args:
            message: User's message text
            message_type: Type of message (user, quick_action, etc.)
            metadata: Additional metadata

        Returns:
            Logged message data
        """
        timestamp = datetime.now()

        message_data = {
            "event_type": "USER_MESSAGE",
            "session_id": self.session_id,
            "timestamp": timestamp.isoformat(),
            "message": message,
            "message_type": message_type,
            "metadata": metadata or {}
        }

        self.log_event(message_data)

        logger.info(f"[CHAT AUDIT] User message logged: {message[:50]}...")

        return message_data

    def log_ai_response(
        self,
        response: str,
        ai_model: str = "DeepSeek",
        response_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log AI response.

        Args:
            response: AI's response text
            ai_model: AI model used
            response_time_ms: Response time in milliseconds
            metadata: Additional metadata

        Returns:
            Logged response data
        """
        timestamp = datetime.now()

        response_data = {
            "event_type": "AI_RESPONSE",
            "session_id": self.session_id,
            "timestamp": timestamp.isoformat(),
            "response": response,
            "ai_model": ai_model,
            "response_time_ms": response_time_ms,
            "metadata": metadata or {}
        }

        self.log_event(response_data)

        logger.info(f"[CHAT AUDIT] AI response logged ({response_time_ms}ms)")

        return response_data

    def log_action_execution(
        action_name: str,
        action_type: str,
        success: bool = True,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Log action execution (e.g., close positions, pause trading).

        Args:
            action_name: Name of the action
            action_type: Type of action (trading, system, etc.)
            success: Whether action succeeded
            result: Action result data
            error: Error message if failed

        Returns:
            Logged action data
        """
        timestamp = datetime.now()

        action_data = {
            "event_type": "ACTION_EXECUTION",
            "session_id": self.session_id,
            "timestamp": timestamp.isoformat(),
            "action_name": action_name,
            "action_type": action_type,
            "success": success,
            "result": result,
            "error": error
        }

        self.log_event(action_data)

        status = "SUCCESS" if success else "FAILED"
        logger.info(f"[CHAT AUDIT] Action {status}: {action_name}")

        return action_data

    def log_conversation_summary(
        self,
        user_message: str,
        ai_response: str,
        context_used: Dict[str, Any],
        tokens_used: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Log complete conversation summary.

        Args:
            user_message: User's message
            ai_response: AI's response
            context_used: Context data used for response
            tokens_used: Number of tokens used

        Returns:
            Logged summary
        """
        timestamp = datetime.now()

        summary = {
            "event_type": "CONVERSATION_SUMMARY",
            "session_id": self.session_id,
            "timestamp": timestamp.isoformat(),
            "user_message": user_message,
            "ai_response": ai_response,
            "context_summary": {
                "active_positions": len(context_used.get("active_positions", {})),
                "total_exposure": context_used.get("risk_metrics", {}).get("total_exposure", 0),
                "current_drawdown": context_used.get("risk_metrics", {}).get("current_drawdown", 0),
                "market_regime": context_used.get("market_regime", "UNKNOWN"),
                "system_health": context_used.get("system_health", {})
            },
            "tokens_used": tokens_used
        }

        self.log_event(summary)

        return summary

    def log_event(self, event_data: Dict[str, Any]) -> None:
        """
        Log event to file and console.

        Args:
            event_data: Event data to log
        """
        try:
            # Write to log file (JSON lines format)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data, default=str) + '\n')

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.

        Args:
            session_id: Session ID to get stats for

        Returns:
            Session statistics
        """
        stats = {
            "session_id": session_id,
            "message_count": 0,
            "ai_response_count": 0,
            "action_count": 0,
            "total_tokens": 0,
            "start_time": None,
            "end_time": None
        }

        try:
            if not self.log_file.exists():
                return stats

            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    event = json.loads(line)

                    if event.get("session_id") != session_id:
                        continue

                    if event["event_type"] == "SESSION_START":
                        stats["start_time"] = event["timestamp"]
                    elif event["event_type"] == "USER_MESSAGE":
                        stats["message_count"] += 1
                    elif event["event_type"] == "AI_RESPONSE":
                        stats["ai_response_count"] += 1
                        stats["total_tokens"] += event.get("metadata", {}).get("tokens_used", 0)
                    elif event["event_type"] == "ACTION_EXECUTION":
                        stats["action_count"] += 1
                    elif event["event_type"] == "SESSION_END":
                        stats["end_time"] = event["timestamp"]

        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")

        return stats

    def export_session_log(self, session_id: str, output_file: Optional[str] = None) -> str:
        """
        Export session log to a human-readable file.

        Args:
            session_id: Session ID to export
            output_file: Output file path (optional)

        Returns:
            Path to exported file
        """
        if not output_file:
            output_file = self.log_dir / f"session_{session_id}.txt"

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Chat Session Log: {session_id}\n")
                f.write(f"=" * 80 + "\n\n")

                if self.log_file.exists():
                    with open(self.log_file, 'r', encoding='utf-8') as log_f:
                        for line in log_f:
                            if not line.strip():
                                continue

                            event = json.loads(line)

                            if event.get("session_id") != session_id:
                                continue

                            timestamp = event.get("timestamp", "")

                            if event["event_type"] == "SESSION_START":
                                f.write(f"\n[{timestamp}] Session Started\n")
                            elif event["event_type"] == "USER_MESSAGE":
                                f.write(f"\n[{timestamp}] USER:\n")
                                f.write(f"  {event.get('message', '')}\n")
                            elif event["event_type"] == "AI_RESPONSE":
                                f.write(f"\n[{timestamp}] DEEPSEEK AI:\n")
                                f.write(f"  {event.get('response', '')}\n")
                            elif event["event_type"] == "ACTION_EXECUTION":
                                f.write(f"\n[{timestamp}] ACTION: {event.get('action_name', '')}\n")
                                f.write(f"  Status: {'SUCCESS' if event.get('success') else 'FAILED'}\n")
                            elif event["event_type"] == "SESSION_END":
                                f.write(f"\n[{timestamp}] Session Ended\n")

            logger.info(f"Session log exported to: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Failed to export session log: {e}")
            return ""

    def end_session(self) -> Dict[str, Any]:
        """
        End the current session.

        Returns:
            Session end event data
        """
        if not self.session_id:
            return {}

        end_time = datetime.now()
        duration = (end_time - self.session_start).total_seconds() if self.session_start else 0

        session_data = {
            "event_type": "SESSION_END",
            "session_id": self.session_id,
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration,
            "start_time": self.session_start.isoformat() if self.session_start else None
        }

        self.log_event(session_data)

        logger.info(f"[CHAT AUDIT] Session ended: {self.session_id} ({duration:.1f}s)")

        return session_data
