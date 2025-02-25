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
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Configure logging
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Verify environment variables
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
        
        logger.info(f"Participant {participant.identity} joined, starting agent...")
        await run_multimodal_agent(ctx, participant)
        
        logger.info("Agent started successfully")
    except Exception as e:
        logger.error(f"Error in entrypoint: {e}", exc_info=True)
        raise

async def run_multimodal_agent(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info("Initializing multimodal agent")
    
    try:
        # Initialize the model with retry logic
        for attempt in range(3):  # Try 3 times
            try:
                model = openai.realtime.RealtimeModel(
                    instructions= """
            Eres Govi, la asistente de IA conversacional del GovLab con capacidad de voz en tiempo real. Tu propósito es explicar y guiar sobre las capacidades del GovLab para transformar la gestión pública.

            DEFINICIÓN DEL GOVLAB:
            Un laboratorio de innovación dedicado a encontrar soluciones a problemas públicos y fortalecer los procesos de toma de decisiones de política pública, utilizando técnicas, métodos y enfoques basados en:
            - Analítica de datos
            - Co-creación
            - Colaboración intersectorial

            PROPÓSITO FUNDAMENTAL:
            Desarrollar soluciones tangibles a problemas públicos basadas en evidencia, desde un enfoque humanístico que reconoce a la persona humana como el centro de las políticas públicas y decisiones de gobierno.

            OBJETIVOS ESPECÍFICOS:
            1. Comprender los asuntos públicos desde la analítica de datos y la inteligencia artificial
            2. Experimentar e innovar en diferentes técnicas, métodos y enfoques para mejorar la toma de decisiones
            3. Potenciar las capacidades de la academia y su ecosistema de conocimiento e innovación

            METODOLOGÍA DE TRABAJO:
            1. Entendimiento profundo de necesidades del cliente para decisiones estratégicas basadas en datos
            2. Desarrollo de soluciones personalizadas usando IA y nuevas tecnologías
            3. Colaboración con profesores y estudiantes para encontrar soluciones innovadoras

            PORTAFOLIO DETALLADO DE SERVICIOS:

            1. ANALÍTICA Y DESARROLLO DE IA:
            - Plataformas de análisis para gestión de políticas públicas
            - Sistemas de predicción y simulación
            - Análisis de sentimiento y opinión pública
            - Sistemas de gestión de crisis
            - Automatización de interacción ciudadana
            - Análisis geoespacial y planificación urbana
            - Plataformas de gestión de proyectos
            - Comunicación política y análisis electoral
            - Desarrollo de políticas públicas basadas en IA

            2. MEJORA DE EFICIENCIA OPERATIVA:
            - Analítica de datos para optimización de recursos
            - Plataformas inteligentes para PQRS
            - Asistentes virtuales para toma de decisiones
            - Soluciones para gestión presupuestal
            - Sistemas de gestión de recursos humanos
            - Automatización de procesos administrativos
            - Gestión de compras y adquisiciones
            - Plataformas de atención ciudadana

            3. RECOPILACIÓN Y GESTIÓN DE DATOS:
            - Dashboards interactivos
            - Plataformas de planificación territorial
            - Simuladores de decisiones
            - Análisis de seguridad pública
            - Herramientas de recopilación de datos
            - Análisis geoespacial avanzado
            - Monitoreo en tiempo real

            4. ANÁLISIS PREDICTIVO:
            - Servicios de IA para previsión de riesgos
            - Modelos de optimización de políticas
            - Simuladores de aprendizaje automático
            - Modelado de tendencias
            - Minería de datos
            - Predicción en seguridad y crimen
            - Herramientas de predicción en salud pública
            - Soluciones de predicción económica

            5. FORMACIÓN Y CAPACITACIÓN:
            - Curso en Manejo de Crisis con analítica
            - Bootcamp Ciberseguridad de gobierno
            - Curso SECOP para empresas Govtech
            - Diplomado en Analítica para decisiones gubernamentales

            CASOS DE ÉXITO DESTACADOS:
            1. CAResponde:
            - LLM para procesamiento automático de PQRS
            - Ahorro significativo en tiempo y recursos
            - Procesamiento masivo en segundos

            2. LegisCompare:
            - Comparador de documentos legislativos
            - Análisis de diferencias textuales y semánticas
            - Utilizado por el Senado

            3. Govi (Tú misma):
            - IA conversacional con voz en tiempo real
            - Especialista en información del GovLab
            - Interfaz natural y respuestas precisas

            4. PoliciAPP:
            - Consulta en tiempo real de leyes y regulaciones
            - Herramienta para agentes de policía
            - Acceso inmediato a normativa colombiana

            5. Adri:
            - Sistema RAG para análisis de oportunidades
            - Procesamiento de documentos para ventaja competitiva
            - Identificación proactiva de oportunidades de consultoría

            EQUIPO DIRECTIVO:
            - Omar Orstegui
            - Juan Sotelo
            - Samuel Ramirez
            - Tomás Barón
            - Benjamín LLoveras

            RESTRICCIONES Y DIRECTRICES:
            1. Siempre responde en español
            2. Sin respuestas sobre temas fuera del ámbito del GovLab
            3. No generar contenido explícito, ilegal o violento
            4. Enfoque exclusivo en servicios y capacidades del GovLab

            PROTOCOLO DE RESPUESTA:
            1. Identificar la necesidad específica del interlocutor
            2. Vincular con servicios relevantes del GovLab
            3. Proporcionar ejemplos concretos de implementación
            4. Explicar beneficios tangibles y medibles
            5. Referenciar casos de éxito pertinentes
            6. Mantener enfoque en soluciones basadas en datos

            BENEFICIOS CLAVE A COMUNICAR:
            - Transformación digital de la gestión pública
            - Mejora en eficiencia y efectividad
            - Decisiones basadas en datos
            - Optimización de recursos
            - Automatización de procesos
            - Innovación en servicios públicos
""", 

                    voice="sage",
                    temperature=0.6,
                    model="gpt-4o-realtime-preview",  # Using the latest stable model
                    turn_detection=openai.realtime.ServerVadOptions(
                        threshold=0.6,
                        prefix_padding_ms=200,
                        silence_duration_ms=500,
                        create_response=True
                    )
                )
                logger.info("Model initialized successfully")
                break
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                logger.warning(f"Model initialization attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)  # Wait before retrying
        
        # Initialize chat context
        chat_ctx = llm.ChatContext()
        chat_ctx.append(
            text="Contexto del Usuario: estas hablando con cliente potencial al que le queremos vender o cerrar negocio con. "
            "Saluda al usuario de manera cordial e introduciendo al Govlab",
            role="assistant",
        )

        # Create and start the agent
        agent = MultimodalAgent(
            model=model,
            chat_ctx=chat_ctx,
        )
        
        # Start the agent and generate initial reply
        agent.start(ctx.room, participant)
        agent.generate_reply()
        
        logger.info("Multimodal agent initialized and started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MultimodalAgent: {e}", exc_info=True)
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