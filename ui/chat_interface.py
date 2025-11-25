"""
Chat Interface
Interactive chat with DeepSeek AI for strategy discussions and system control.
Real-time integration with DeepSeek AI with comprehensive audit logging.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List
import dash
from dash import dcc, html, Input, Output, State, callback_context, callback
from loguru import logger

# Import DeepSeek client and system context
import sys

from core.system_context import SystemContext
from deepseek.client import DeepSeekBrain
from ui.chat_audit_logger import ChatAuditLogger


# Global instances (initialized in create_chat_app)
SYSTEM_CONTEXT = None
DEEPSEEK_BRAIN = None
AUDIT_LOGGER = None


# Note: This module provides chat interface components
# Multi-page app registration happens in create_chat_app()


def create_chat_message(message_data: Dict[str, Any]) -> html.Div:
    """
    Create a chat message component.

    Args:
        message_data: Message data with 'user', 'ai', 'timestamp', 'type'

    Returns:
        HTML div for the message
    """
    is_user = message_data.get('is_user', True)

    if is_user:
        # User message (right-aligned)
        return html.Div([
            html.Div([
                html.Div("You", style={
                    'fontWeight': 'bold',
                    'marginBottom': '5px',
                    'color': '#00ff88'
                }),
                html.Div(
                    message_data.get('text', ''),
                    style={
                        'backgroundColor': '#1a1a1a',
                        'padding': '10px',
                        'borderRadius': '8px',
                        'maxWidth': '70%',
                        'marginLeft': 'auto'
                    }
                ),
                html.Div(
                    message_data.get('timestamp', ''),
                    style={
                        'fontSize': '10px',
                        'color': '#666',
                        'marginTop': '5px',
                        'textAlign': 'right'
                    }
                )
            ], style={'textAlign': 'right', 'marginBottom': '15px'})
        ])

    else:
        # AI message (left-aligned)
        return html.Div([
            html.Div([
                html.Div("DeepSeek AI", style={
                    'fontWeight': 'bold',
                    'marginBottom': '5px',
                    'color': '#4488ff'
                }),
                html.Div(
                    message_data.get('text', ''),
                    style={
                        'backgroundColor': '#1a1a1a',
                        'padding': '15px',
                        'borderRadius': '8px',
                        'maxWidth': '70%',
                        'borderLeft': '3px solid #4488ff'
                    }
                ),
                html.Div(
                    message_data.get('timestamp', ''),
                    style={
                        'fontSize': '10px',
                        'color': '#666',
                        'marginTop': '5px'
                    }
                )
            ], style={'marginBottom': '15px'})
        ])


def create_quick_actions() -> html.Div:
    """
    Create quick action buttons.

    Returns:
        HTML div with quick action buttons
    """
    actions = [
        ("📊 Analyze Performance", "analyze_performance"),
        ("🎯 Market Analysis", "market_analysis"),
        ("⚠️ Risk Assessment", "risk_assessment"),
        ("🔧 System Optimization", "system_optimization"),
        ("📈 Feature Performance", "feature_performance"),
        ("💰 Close All Positions", "close_all_positions"),
        ("⏸️ Pause Trading", "pause_trading"),
        ("▶️ Resume Trading", "resume_trading"),
        ("🎯 Run Convergence Strategy", "run_convergence_strategy")
    ]

    buttons = []
    for label, action_id in actions:
        buttons.append(
            html.Button(
                label,
                id={'type': 'quick-action', 'index': action_id},
                n_clicks=0,
                style={
                    'backgroundColor': '#2a2a2a',
                    'color': 'white',
                    'border': '1px solid #444',
                    'padding': '8px 12px',
                    'margin': '5px',
                    'cursor': 'pointer',
                    'borderRadius': '5px',
                    'fontSize': '14px'
                }
            )
        )

    return html.Div([
        html.H4("Quick Actions", style={'color': '#00ff88', 'marginBottom': '10px'}),
        html.Div(buttons, style={'display': 'flex', 'flexWrap': 'wrap'})
    ])


def layout():
    """Chat interface layout."""
    return html.Div([
        # Header
        html.Div([
            html.H1("DeepSeek AI Chat", style={'margin': '0', 'color': '#00ff88'}),
            html.P("Interactive strategy discussions and system control", style={'margin': '5px 0', 'color': '#888'})
        ], style={'padding': '20px', 'textAlign': 'center'}),

        # Quick Actions
        html.Div(
            id='quick-actions-container',
            children=create_quick_actions(),
            style={'padding': '20px', 'backgroundColor': '#1a1a1a', 'marginBottom': '20px'}
        ),

        # Chat Container
        html.Div([
            # Chat history
            html.Div(
                id='chat-history',
                children=[
                    # Welcome message
                    create_chat_message({
                        'is_user': False,
                        'text': 'Hello! I am DeepSeek, your AI trading assistant. I can help you analyze markets, review performance, assess risks, and optimize your trading system. How can I help you today?',
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'type': 'welcome'
                    })
                ],
                style={
                    'height': '500px',
                    'overflowY': 'scroll',
                    'padding': '20px',
                    'backgroundColor': '#0a0a0a',
                    'borderRadius': '8px',
                    'border': '1px solid #333',
                    'marginBottom': '20px'
                }
            ),

            # Input area
            html.Div([
                dcc.Textarea(
                    id='chat-input',
                    placeholder='Type your message to DeepSeek...',
                    style={
                        'width': '100%',
                        'height': '80px',
                        'backgroundColor': '#1a1a1a',
                        'color': 'white',
                        'border': '1px solid #444',
                        'borderRadius': '5px',
                        'padding': '10px',
                        'fontSize': '14px',
                        'resize': 'vertical'
                    }
                ),
                html.Div([
                    html.Button(
                        'Send Message',
                        id='send-button',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#00ff88',
                            'color': 'black',
                            'border': 'none',
                            'padding': '10px 30px',
                            'cursor': 'pointer',
                            'borderRadius': '5px',
                            'fontSize': '16px',
                            'fontWeight': 'bold',
                            'marginTop': '10px'
                        }
                    ),
                    html.Button(
                        'Clear Chat',
                        id='clear-button',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#444',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 30px',
                            'cursor': 'pointer',
                            'borderRadius': '5px',
                            'fontSize': '16px',
                            'marginLeft': '10px',
                            'marginTop': '10px'
                        }
                    )
                ], style={'textAlign': 'right'}),

                # Chat status with audit logging info
                html.Div(
                    id='chat-status',
                    children='Audit logging enabled - logs/chat_audit_*.log',
                    style={
                        'color': '#888',
                        'fontSize': '12px',
                        'marginTop': '10px',
                        'textAlign': 'left'
                    }
                )
            ], style={'backgroundColor': '#1a1a1a', 'padding': '20px', 'borderRadius': '8px'})

        ], style={'maxWidth': '1200px', 'margin': '0 auto'}),

        # Auto-refresh for new messages
        dcc.Interval(
            id='chat-interval',
            interval=3000,  # Check every 3 seconds
            n_intervals=0
        ),

        # Store for chat history
        dcc.Store(id='chat-store', data=[]),
        # Store for chat-to-chart interactions
        dcc.Store(id='chat-highlight-store', data={})

    ], style={
        'backgroundColor': '#0a0a0a',
        'minHeight': '100vh',
        'padding': '20px',
        'color': 'white',
        'fontFamily': 'Arial, sans-serif'
    })


# Helper function to get current system context
def get_system_context_dict() -> Dict[str, Any]:
    """Get system context as a dictionary for DeepSeek."""
    if not SYSTEM_CONTEXT:
        return {}

    return {
        "active_positions": SYSTEM_CONTEXT.active_positions,
        "risk_metrics": SYSTEM_CONTEXT.risk_metrics,
        "market_regime": SYSTEM_CONTEXT.market_regime,
        "system_health": SYSTEM_CONTEXT.system_health,
        "trade_history_count": len(SYSTEM_CONTEXT.trade_history),
        "feature_performance": {
            name: {
                "accuracy": metrics.accuracy,
                "total_signals": metrics.total_signals,
                "trend": metrics.trend
            }
            for name, metrics in SYSTEM_CONTEXT.feature_performance.items()
        }
    }


# Helper function to call DeepSeek async
def call_deepseek_chat(user_message: str, message_type: str = "strategy") -> str:
    """
    Call DeepSeek chat interface.

    Args:
        user_message: User's message
        message_type: Type of message

    Returns:
        AI response
    """
    if not DEEPSEEK_BRAIN or not SYSTEM_CONTEXT:
        return "DeepSeek AI is not initialized. Please check system configuration."

    try:
        # Get system context
        context = get_system_context_dict()

        # Run async call in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(
            DEEPSEEK_BRAIN.chat_interface(user_message, context, message_type)
        )
        loop.close()

        return response

    except Exception as e:
        error_msg = f"Error calling DeepSeek: {str(e)}"
        logger.error(error_msg)
        return error_msg


# Callbacks
@callback(
    [Output('chat-store', 'data'),
     Output('chat-history', 'children')],
    [Input('send-button', 'n_clicks'),
     Input('clear-button', 'n_clicks'),
     Input({'type': 'quick-action', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('chat-input', 'value'),
     State('chat-store', 'data')]
)
def update_chat(send_clicks, clear_clicks, action_clicks, input_text, chat_history):
    """Update chat history with real DeepSeek responses."""
    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Initialize audit logger if needed
    global AUDIT_LOGGER
    if AUDIT_LOGGER is None:
        AUDIT_LOGGER = ChatAuditLogger()
        AUDIT_LOGGER.start_session()

    # Clear chat
    if trigger and 'clear-button' in trigger and clear_clicks > 0:
        AUDIT_LOGGER.log_event({
            "event_type": "CHAT_CLEARED",
            "timestamp": datetime.now().isoformat()
        })

        return [], [create_chat_message({
            'is_user': False,
            'text': 'Chat cleared. How can I help you?',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })]

    # Send message
    if trigger and 'send-button' in trigger and send_clicks > 0 and input_text:
        # Add user message
        user_message = {
            'is_user': True,
            'text': input_text,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'user'
        }

        # Log user message
        AUDIT_LOGGER.log_user_message(input_text, "user")

        # Call DeepSeek for real AI response
        start_time = datetime.now()
        ai_response_text = call_deepseek_chat(input_text, "strategy")
        response_time = (datetime.now() - start_time).total_seconds() * 1000

        ai_response = {
            'is_user': False,
            'text': ai_response_text,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'ai',
            'response_time_ms': response_time
        }

        # Log AI response with audit
        AUDIT_LOGGER.log_ai_response(
            ai_response_text,
            response_time_ms=response_time,
            metadata={"context_used": True}
        )

        # Log conversation summary
        AUDIT_LOGGER.log_conversation_summary(
            input_text,
            ai_response_text,
            get_system_context_dict()
        )

        chat_history = chat_history + [user_message, ai_response]

        # Update chat history display
        messages = chat_history[-50:]  # Keep last 50 messages
        message_components = [create_chat_message(msg) for msg in messages]

        return chat_history, message_components

    # Quick action
    if trigger and isinstance(trigger, dict) and 'type' in trigger:
        # Get action ID
        action_id = trigger['index']

        # Predefined messages for quick actions
        action_messages = {
            'analyze_performance': 'Please analyze our recent trading performance and suggest improvements.',
            'market_analysis': 'Provide a deep analysis of current market conditions and opportunities.',
            'risk_assessment': 'Evaluate our current risk exposure and suggest management strategies.',
            'system_optimization': 'Identify system bottlenecks and suggest performance improvements.',
            'feature_performance': 'Review current feature effectiveness and recommend adjustments.',
            'close_all_positions': 'Please close all open positions immediately.',
            'pause_trading': 'Pause all trading activities.',
            'resume_trading': 'Resume normal trading operations.',
            'run_convergence_strategy': 'Please run the Multi-Timeframe Convergence Strategy and provide a detailed analysis of the current signal, including alignment score, market regime, and satisfied conditions.'
        }

        message = action_messages.get(action_id, f'Action: {action_id}')

        # Add user message
        user_message = {
            'is_user': True,
            'text': message,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'quick_action',
            'action_id': action_id
        }

        # Log user message
        AUDIT_LOGGER.log_user_message(message, "quick_action", {"action_id": action_id})

        # Call DeepSeek for action-specific response
        start_time = datetime.now()
        ai_response_text = call_deepseek_chat(message, "system_action")
        response_time = (datetime.now() - start_time).total_seconds() * 1000

        ai_response = {
            'is_user': False,
            'text': ai_response_text,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'ai',
            'response_time_ms': response_time
        }

        # Log AI response
        AUDIT_LOGGER.log_ai_response(
            ai_response_text,
            response_time_ms=response_time,
            metadata={"quick_action": action_id}
        )

        chat_history = chat_history + [user_message, ai_response]

        # Update chat history display
        messages = chat_history[-50:]
        message_components = [create_chat_message(msg) for msg in messages]

        return chat_history, message_components

    # Return current state
    messages = chat_history[-50:] if chat_history else []
    message_components = [create_chat_message(msg) for msg in messages]

    return chat_history, message_components


# Auto-scroll callback
@callback(
    Output('chat-history', 'scrollPosition'),
    [Input('chat-store', 'data')]
)
def auto_scroll(chat_history):
    """Auto-scroll to bottom of chat."""
    return {'vertical': 'bottom'}


# Chat-to-chart interaction callback
@callback(
    Output('chat-highlight-store', 'data'),
    [Input('chat-history', 'children')],
    [State('chat-store', 'data')]
)
def highlight_chart_from_chat(chat_children, chat_history):
    """
    Extract overlay references from chat messages and provide highlights for chart.
    This enables chat commands to affect chart overlays.
    """
    if not chat_history or len(chat_history) < 2:
        return {}

    # Get the latest AI message
    latest_message = chat_history[-1] if chat_history else None
    if not latest_message or latest_message.get('is_user', True):
        return {}

    message_text = latest_message.get('text', '').lower()

    # Map chat keywords to chart features
    overlay_mapping = {
        'liquidity': 'liquidity',
        'liquid zones': 'liquidity',
        'supertrend': 'supertrend',
        'trend': 'supertrend',
        'chandelier': 'chandelier',
        'chandelier exit': 'chandelier',
        'order flow': 'orderflow',
        'flow': 'orderflow',
        'regime': 'regime',
        'market regime': 'regime',
        'alignment': 'alignment',
        'timeframe alignment': 'alignment'
    }

    # Check which overlays are mentioned in the message
    highlighted_features = []
    for keyword, feature in overlay_mapping.items():
        if keyword in message_text:
            highlighted_features.append(feature)

    # If specific features mentioned, return highlight data
    if highlighted_features:
        return {
            'highlighted_overlays': highlighted_features,
            'timestamp': datetime.now().isoformat(),
            'message_index': len(chat_history) - 1,
            'action': 'highlight'
        }

    return {}


def initialize_deepseek():
    """
    Initialize DeepSeek brain and system context.

    Returns:
        Tuple of (DeepSeekBrain, SystemContext) or (None, None) if initialization fails
    """
    try:
        # Initialize system context
        system_context = SystemContext()

        # Try to initialize DeepSeek brain
        # If it fails (e.g., missing API key), return None for DeepSeek but keep SystemContext
        try:
            deepseek_brain = DeepSeekBrain(system_context)
            logger.info("DeepSeek AI initialized successfully for chat interface")
            return deepseek_brain, system_context
        except Exception as e:
            logger.warning(f"Failed to initialize DeepSeek AI: {e}")
            logger.warning("Chat interface will work in limited mode without AI responses")
            return None, system_context

    except Exception as e:
        logger.error(f"Failed to initialize system context: {e}")
        return None, None


def create_chat_app():
    """Create the chat Dash app with DeepSeek integration."""
    global DEEPSEEK_BRAIN, SYSTEM_CONTEXT, AUDIT_LOGGER

    # Initialize DeepSeek and system context
    DEEPSEEK_BRAIN, SYSTEM_CONTEXT = initialize_deepseek()

    if SYSTEM_CONTEXT:
        logger.info("Chat interface initialized with system context")
    else:
        logger.warning("Chat interface initialized without system context")

    if DEEPSEEK_BRAIN:
        logger.info("Chat interface initialized with DeepSeek AI")
    else:
        logger.warning("Chat interface running without DeepSeek AI (demo mode)")

    # Initialize audit logger
    AUDIT_LOGGER = ChatAuditLogger()
    AUDIT_LOGGER.start_session()

    # Create Dash app
    app = dash.Dash(__name__)
    app.layout = layout

    # Add audit logging callback
    @app.callback(
        Output('chat-status', 'children'),
        [Input('chat-store', 'data')]
    )
    def update_chat_status(chat_history):
        """Update chat status showing audit logging info."""
        if AUDIT_LOGGER and AUDIT_LOGGER.session_id:
            return f"Audit logging enabled - Session: {AUDIT_LOGGER.session_id} - Logs: logs/chat_audit_{datetime.now().strftime('%Y%m%d')}.log"
        return "Audit logging enabled - logs/chat_audit_*.log"

    return app
