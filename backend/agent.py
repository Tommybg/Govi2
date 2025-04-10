from __future__ import annotations

import logging
import os
import asyncio
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    AgentSession,
)
from livekit.plugins import openai

# Cargar variables de entorno
load_dotenv(dotenv_path=".env.local")

# Configurar logging
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

# Verificar variables de entorno requeridas
required_env_vars = ['OPENAI_API_KEY', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET']
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

async def entrypoint(ctx: JobContext):
    try:
        logger.info(f"Connecting to room {ctx.room.name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        logger.info("Waiting for participant...")
        participant = await ctx.wait_for_participant()

        logger.info(f"Participant {participant.identity} joined, starting agent session...")
        await run_agent_session(ctx, participant)

        logger.info("Agent session started successfully")
    except Exception as e:
        logger.error(f"Error in entrypoint: {e}", exc_info=True)
        raise

async def run_agent_session(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info("Initializing agent session")

    try:
        # Inicializar el modelo en tiempo real con retry logic (3 intentos)
        for attempt in range(3):
            try:
                realtime_model = openai.realtime.RealtimeModel(
                    instructions="""
Eres Govi, la asistente de IA conversacional del GovLab con capacidad de voz en tiempo real. Tu propósito es explicar y guiar sobre las capacidades del GovLab para transformar la gestión pública.

DEFINICIÓN DEL GOVLAB:
Un laboratorio de innovación dedicado a encontrar soluciones a problemas públicos y fortalecer los procesos de toma de decisiones de política pública, utilizando técnicas, métodos y enfoques basados en:
- Analítica de datos
- Co-creación
- Colaboración intersectorial

PROPÓSITO FUNDAMENTAL:
Desarrollar soluciones tangibles a problemas públicos basadas en evidencia, desde un enfoque humanístico que reconoce a la persona humana como el centro de las políticas públicas y decisiones de gobierno.

(Se omiten detalles adicionales para mayor claridad)
""",
                    voice="echo",
                    temperature=0.6,
                    model="gpt-4o-realtime-preview"  # Usando el modelo estable actual
                    # Se elimina el parámetro turn_detection ya que ServerVadOptions ya no existe
                )
                logger.info("Realtime model initialized successfully")
                break
            except Exception as e:
                if attempt == 2:  # Último intento
                    raise
                logger.warning(f"Model initialization attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        
        # Crear el contexto de chat y agregar un mensaje de sistema
        chat_ctx = llm.ChatContext()
        chat_ctx.append(
            role="assistant",
            text="Contexto del Usuario: estas hablando con un cliente potencial. Saluda al usuario de manera cordial e introduce al GovLab."
        )

        # Crear la sesión del agente usando AgentSession, pasando el modelo en tiempo real
        session = AgentSession(
            llm=realtime_model,
            chat_ctx=chat_ctx,
        )
        
        # Iniciar la sesión del agente en la sala y para el participante
        await session.start(ctx.room, participant)
        # Enviar el mensaje inicial
        await session.send("Hola, ¿en qué puedo ayudarte hoy?", allow_interruptions=True)
        
        logger.info("Agent session initialized and started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize AgentSession: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
            )
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise

