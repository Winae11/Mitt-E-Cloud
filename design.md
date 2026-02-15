# Mitt e Cloud - High-Level Design

## System Architecture Overview
Fully serverless, edge-aware architecture on AWS optimized for rural scalability, low latency, and intermittent connectivity.

- Data Sources → Ingestion → AI Prediction → Conversation Layer → Delivery → Feedback & Rewards

## Key Components
1. Data Ingestion Layer
   - Public APIs: IMD weather JSON, ISRO Bhuvan satellite (NDVI/soil moisture), public soil databases
   - Optional: Farmer inputs (voice notes, simple text/photos via WhatsApp)

2. Prediction Engine (Core AI)
   - Amazon Bedrock: Multimodal foundation models (e.g., Claude/Titan)
     - Processes satellite imagery + time-series weather + crop params
     - Computes sustainability scores (water savings %, methane reduction potential) using RAG + custom prompts
     - Generates region/crop-specific recommendations (e.g., AWD schedules for rice fields)
   - Scheduled daily/ hourly triggers via AWS Lambda

3. Conversational & Delivery Layer
   - Amazon Q: Dialect-aware conversational AI with ASR (speech recognition) and TTS (text-to-speech)
     - Supports Hindi, Marathi, Punjabi, and other regional variants
     - Handles follow-up queries with context (e.g., "Aur kitna pani dalun?")
   - Delivery Channels: WhatsApp Business API (primary), Twilio/Pinpoint SMS (fallback for feature phones)

4. Offline / Edge Support
   - On-device caching: Lightweight predictions stored locally (via progressive web app or simple SMS cache)
   - Community swarm: Broadcast via WhatsApp groups or SMS chains when thresholds met

5. Data Storage & Orchestration
   - AWS S3: Secure, anonymized storage for inputs and model outputs
   - AWS Glue: ETL pipelines for incoming feeds
   - DynamoDB: Gamification state, action logs, reward balances
   - AWS Lambda: Event-driven orchestration for forecasts and alerts

6. Security & Scalability
   - Encryption: AWS KMS for all data at rest/transit
   - Access Control: IAM roles with least privilege
   - Auto-scaling: Handles seasonal peaks (e.g., monsoon/heatwave alerts)

## High-Level Flow
1. Cron/scheduled → Pull IMD + satellite data → Bedrock inference → Store predictions
2. Farmer interacts (voice/text/photo) → Q processes query → Matches to prediction → Delivers advisory
3. Action confirmation → Logs to DynamoDB → Updates rewards → Refines future suggestions

## Tech Choices Rationale
- Bedrock + Q: Required for meaningful GenAI – predictive sustainability modeling + natural, inclusive conversations
- Serverless: No infrastructure management, cost-effective for variable rural demand
- WhatsApp/SMS-first: Matches rural penetration (no app install needed, works on basic phones)
- Offline/swarm focus: Addresses Bharat's connectivity gaps while building community resilience