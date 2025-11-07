# Global Architecture Document (GAD)
## ADOR - Augmented Document Reader System

**Version:** 1.0
**Date:** 07-11-2025
**Author:** Jasir Mohammed

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture Components](#3-architecture-components)
4. [Component Details](#4-component-details)
5. [Data Flow](#5-data-flow)
6. [API Design](#6-api-design)
7. [Security Architecture](#7-security-architecture)
8. [Document Processing Pipeline](#8-document-processing-pipeline)
9. [Storage and Database](#9-storage-and-database)
10. [Logging and Monitoring](#10-logging-and-monitoring)
11. [Error Handling and Fallback Mechanisms](#11-error-handling-and-fallback-mechanisms)
12. [Scalability and Performance](#12-scalability-and-performance)
13. [Deployment Architecture](#13-deployment-architecture)
14. [Integration with CMI IS](#14-integration-with-cmi-is)
15. [Technology Stack](#15-technology-stack)

---

## 1. Executive Summary

The Augmented Document Reader (ADOR) is a production-level system designed to process financial documents and extract structured information using AI techniques. The system handles three types of documents - structured DOCX files, informal chat messages, and unstructured PDF documents - using different processing strategies for each.

The architecture is designed for:
- High availability and fault tolerance
- Secure handling of confidential financial data
- Real-time and batch processing capabilities
- Integration with existing CMI Information Systems
- Comprehensive logging and monitoring
- Horizontal scalability

---

## 2. System Overview

### 2.1 Core Capabilities

The ADOR system provides the following features:

1. **Document Classification** - Identify document type and route to appropriate processor
2. **Named Entity Recognition (NER)** - Extract financial entities from documents
3. **Document Summarization** - Generate concise summaries of lengthy documents
4. **Topic Modeling** - Identify key topics and themes in documents
5. **Question Answering** - Answer specific questions about document content

### 2.2 Supported Document Types

| Document Type | Examples | Processing Method | Expected Throughput |
|---------------|----------|-------------------|---------------------|
| Structured DOCX | Term sheets, contracts | Rule-based parser | 100 docs/minute |
| Chat Messages | Trading chats, Slack/Teams | NER model (spaCy) | 500 messages/minute |
| Unstructured PDF | Research reports, presentations | LLM with RAG | 20 docs/minute |

### 2.3 High-Level Architecture

```
[External Systems]                [ADOR System]                    [Storage]
     |                                  |                              |
     |                                  |                              |
CMI IS --> API Gateway --> Load Balancer --> Document Service --> Database
  Apps        |                |                    |                |
              |                |                    |                |
 Web UI ------+                +-------> Processing Queue            |
              |                              |    |    |             |
         Auth Service                        |    |    |             |
              |                          Parser NER  LLM             |
              |                              |    |    |             |
         ACL Service                         +----+----+             |
                                                  |                  |
                                           Result Store ------------>+
                                                  |
                                           Monitoring/Logs
```

---

## 3. Architecture Components

### 3.1 Component List

The system consists of the following major components:

1. **API Gateway** - Entry point for all requests, handles authentication
2. **Load Balancer** - Distributes traffic across multiple service instances
3. **Document Service** - Main orchestrator for document processing
4. **Authentication Service** - Manages user authentication and tokens
5. **Authorization Service (ACL)** - Controls access to documents and features
6. **Document Classifier** - Determines document type and routing
7. **Processing Queue** - Message queue for asynchronous processing
8. **Parser Service** - Handles structured DOCX files
9. **NER Service** - Processes chat messages with NER models
10. **LLM Service** - Handles unstructured PDFs with language models
11. **Storage Service** - Manages document storage and retrieval
12. **Database** - Stores metadata, results, and user data
13. **Cache Layer** - Redis for temporary data and session management
14. **Logging Service** - Centralized logging and audit trails
15. **Monitoring Service** - System health and performance monitoring
16. **Web UI** - User interface for document upload and interaction

---

## 4. Component Details

### 4.1 API Gateway

**Purpose:** Central entry point for all API requests, provides security and routing.

**Technology:** Kong or AWS API Gateway

**Responsibilities:**
- Request validation and sanitization
- Rate limiting (1000 requests per minute per user)
- Authentication token verification
- Request routing to appropriate services
- SSL/TLS termination
- API versioning support

**Configuration:**
```yaml
api_gateway:
  host: api.ador.cmi.com
  port: 443
  rate_limit:
    user: 1000/minute
    ip: 5000/minute
  timeout: 30s
  max_body_size: 100MB
```

### 4.2 Load Balancer

**Purpose:** Distribute incoming requests across multiple service instances.

**Technology:** NGINX or AWS Application Load Balancer

**Features:**
- Round-robin distribution
- Health checks every 10 seconds
- Automatic removal of unhealthy instances
- Session persistence (sticky sessions)
- WebSocket support for real-time features

**Health Check Endpoint:**
```
GET /health
Response: {"status": "healthy", "version": "1.0.0", "uptime": 3600}
```

### 4.3 Document Service

**Purpose:** Main orchestrator that coordinates document processing workflow.

**Technology:** Python FastAPI

**Key Functions:**

1. **Document Upload**
   - Receives documents from API or UI
   - Validates file type and size
   - Stores in blob storage
   - Creates database record
   - Initiates processing

2. **Processing Coordination**
   - Sends document to classifier
   - Routes to appropriate processor based on classification
   - Manages processing status
   - Handles callbacks from processors

3. **Result Aggregation**
   - Collects results from various processors
   - Formats output according to client preferences
   - Caches results for repeat requests

**API Endpoints:**
```
POST   /api/v1/documents/upload
GET    /api/v1/documents/{id}
GET    /api/v1/documents/{id}/status
POST   /api/v1/documents/{id}/process
GET    /api/v1/documents/{id}/results
DELETE /api/v1/documents/{id}
```

### 4.4 Authentication Service

**Purpose:** Manage user authentication and generate access tokens.

**Technology:** OAuth 2.0 with JWT tokens

**Authentication Flow:**
```
1. User submits credentials (username/password or SSO)
2. Service validates against Active Directory or LDAP
3. Generate JWT token with user claims
4. Return token to client (valid for 1 hour)
5. Client includes token in Authorization header for subsequent requests
```

**Token Structure:**
```json
{
  "sub": "user123",
  "name": "John Doe",
  "email": "john.doe@cmi.com",
  "roles": ["trader", "analyst"],
  "permissions": ["read:documents", "write:documents"],
  "exp": 1704672000,
  "iat": 1704668400
}
```

**Security Features:**
- Token refresh mechanism (refresh tokens valid for 7 days)
- Token revocation on logout
- Multi-factor authentication (MFA) support
- Password hashing with bcrypt (cost factor 12)
- Account lockout after 5 failed attempts

### 4.5 Authorization Service (ACL)

**Purpose:** Control access to documents and features based on user permissions.

**Technology:** Casbin or custom RBAC implementation

**Access Control Model:**

1. **Role-Based Access Control (RBAC)**
   - Roles: Admin, Manager, Trader, Analyst, Viewer
   - Each role has specific permissions

2. **Document-Level Permissions**
   - Owner: Full access to document
   - Editor: Can modify and process
   - Viewer: Read-only access
   - No Access: Cannot see document

3. **Confidentiality Levels**
   - Public: All authenticated users
   - Internal: Employees only
   - Confidential: Specific teams only
   - Highly Confidential: Named individuals only

**Permission Check Flow:**
```python
def check_access(user_id, document_id, action):
    # 1. Get user roles
    user_roles = get_user_roles(user_id)

    # 2. Get document confidentiality level
    doc_level = get_document_level(document_id)

    # 3. Check if user has required role for action
    required_permission = f"{action}:documents:{doc_level}"

    # 4. Verify permission
    if has_permission(user_roles, required_permission):
        return True

    # 5. Check document-specific permissions
    return has_document_permission(user_id, document_id, action)
```

**ACL Database Schema:**
```sql
CREATE TABLE user_roles (
    user_id VARCHAR(50),
    role VARCHAR(50),
    granted_at TIMESTAMP,
    expires_at TIMESTAMP,
    PRIMARY KEY (user_id, role)
);

CREATE TABLE document_permissions (
    document_id UUID,
    user_id VARCHAR(50),
    permission VARCHAR(20), -- owner, editor, viewer
    granted_by VARCHAR(50),
    granted_at TIMESTAMP,
    PRIMARY KEY (document_id, user_id)
);

CREATE TABLE role_permissions (
    role VARCHAR(50),
    permission VARCHAR(100),
    PRIMARY KEY (role, permission)
);
```

### 4.6 Document Classifier

**Purpose:** Automatically identify document type to route to correct processor.

**Technology:** Machine learning classifier (scikit-learn) or rule-based

**Classification Logic:**

1. **File Extension Check**
   - .docx -> Structured Parser
   - .txt -> NER Service
   - .pdf -> LLM Service

2. **Content Analysis**
   - Check for table structures (DOCX indicator)
   - Check for chat patterns like timestamps (Chat indicator)
   - Check for paragraphs and sections (PDF indicator)

3. **Confidence Scoring**
   - Each classifier provides confidence score (0-1)
   - If confidence < 0.8, flag for manual review

**Response Format:**
```json
{
  "document_id": "doc-12345",
  "classification": "structured_docx",
  "confidence": 0.95,
  "processor": "parser_service",
  "estimated_time": 5
}
```

### 4.7 Processing Queue

**Purpose:** Manage asynchronous document processing tasks.

**Technology:** RabbitMQ or AWS SQS

**Queue Types:**

1. **High Priority Queue** - Real-time processing (< 10 seconds)
2. **Normal Queue** - Standard processing (< 1 minute)
3. **Batch Queue** - Bulk processing (no time limit)
4. **Retry Queue** - Failed tasks for retry

**Message Format:**
```json
{
  "message_id": "msg-67890",
  "document_id": "doc-12345",
  "user_id": "user123",
  "priority": "high",
  "processor": "ner_service",
  "retry_count": 0,
  "max_retries": 3,
  "timestamp": "2025-01-07T10:30:00Z",
  "metadata": {
    "original_filename": "chat_20250107.txt",
    "file_size": 2048,
    "confidentiality": "internal"
  }
}
```

**Processing Guarantees:**
- At-least-once delivery
- Message deduplication using message_id
- Dead letter queue for failed messages after max retries
- Visibility timeout: 5 minutes

### 4.8 Parser Service

**Purpose:** Process structured DOCX files using rule-based extraction.

**Technology:** Python with python-docx library

**Processing Steps:**

1. Download document from storage
2. Load DOCX file into memory
3. Extract tables and text
4. Apply field mapping rules
5. Extract 9 financial entities
6. Validate extracted data
7. Store results in database
8. Send success notification

**Scalability:**
- Stateless service (can run multiple instances)
- Processing time: 2-5 seconds per document
- Can handle 100 documents per minute per instance

**Fallback Strategy:**
- If table extraction fails, try text-based extraction
- If specific field missing, mark as null (not error)
- Log all extraction attempts for debugging

### 4.9 NER Service

**Purpose:** Process chat messages using Named Entity Recognition models.

**Technology:** spaCy with custom fine-tuned models

**Model Management:**

1. **Model Versions**
   - v1.0: Base spaCy en_core_web_sm
   - v1.1: Fine-tuned on 500 examples (current production)
   - v1.2: Fine-tuned on 1000 examples (in testing)

2. **Model Loading**
   - Models loaded at service startup
   - Kept in memory for fast inference
   - Automatic reload on version update

3. **A/B Testing**
   - Route 10% of traffic to new model version
   - Compare accuracy metrics
   - Gradual rollout if metrics improve

**Processing Steps:**

1. Receive chat message from queue
2. Preprocess text (normalize, clean)
3. Run spaCy NER model
4. Apply custom regex patterns
5. Merge results from both approaches
6. Validate entity types
7. Store results
8. Send notification

**Performance:**
- Processing time: < 200ms per message
- Throughput: 500 messages per minute per instance
- Model size: 50MB in memory

### 4.10 LLM Service

**Purpose:** Process unstructured PDF documents using Large Language Models.

**Technology:** OpenAI GPT-4 or Claude API with RAG (Retrieval-Augmented Generation)

**Architecture:**

```
PDF Document
    |
    v
PDF Parser (PyPDF2/pdfplumber)
    |
    v
Text Extraction
    |
    v
Text Chunker (512 tokens per chunk)
    |
    v
Vector Embeddings (OpenAI text-embedding-ada-002)
    |
    v
Vector Store (Pinecone/FAISS)
    |
    v
Query Processing
    |
    v
Retrieve Relevant Chunks (top 5)
    |
    v
LLM Prompt with Context
    |
    v
Entity Extraction
    |
    v
Structured Output
```

**RAG Implementation:**

1. **Document Chunking**
   - Split PDF into chunks of 512 tokens
   - Maintain 50 token overlap between chunks
   - Preserve paragraph boundaries

2. **Embedding Generation**
   - Use OpenAI text-embedding-ada-002
   - 1536-dimensional vectors
   - Cost: $0.0001 per 1000 tokens

3. **Vector Storage**
   - Store in Pinecone vector database
   - Index by document_id
   - Support semantic search

4. **Retrieval**
   - Query: "Extract financial entities"
   - Retrieve top 5 most relevant chunks
   - Pass to LLM with extraction prompt

**LLM Prompting Strategy:**

```python
ENTITY_EXTRACTION_PROMPT = """
You are a financial document analyzer. Extract the following entities from the text:

Required Entities:
- Counterparty: Name of the bank or financial institution
- Notional: Transaction amount
- Maturity: Time to maturity
- Underlying: Reference asset or security
- Coupon: Interest rate or coupon percentage
- Barrier: Barrier level (if applicable)

Context:
{retrieved_chunks}

Extract entities in JSON format. If an entity is not found, use null.

Response format:
{
  "Counterparty": "...",
  "Notional": "...",
  ...
}
"""
```

**Cost Management:**
- Cache LLM responses for identical queries
- Use streaming for long documents
- Implement token limits (max 4000 tokens per request)
- Estimated cost: $0.03 per PDF document

**Fallback Strategy:**
- If LLM API fails, retry with exponential backoff
- After 3 retries, fall back to NER model
- If both fail, mark document for manual review

### 4.11 Storage Service

**Purpose:** Manage document storage and retrieval.

**Technology:** AWS S3 or Azure Blob Storage

**Storage Structure:**

```
ador-documents/
├── raw/                      # Original uploaded documents
│   ├── 2025/
│   │   ├── 01/
│   │   │   ├── 07/
│   │   │   │   ├── doc-12345.docx
│   │   │   │   ├── doc-12346.txt
│   │   │   │   └── doc-12347.pdf
├── processed/                # Processed documents with annotations
│   └── [same structure]
└── results/                  # Extracted results in JSON
    └── [same structure]
```

**Storage Policies:**

1. **Retention**
   - Raw documents: 7 years (regulatory requirement)
   - Processed documents: 1 year
   - Results: 7 years

2. **Lifecycle Management**
   - Documents older than 90 days move to cold storage
   - Documents older than 1 year archived to glacier
   - Automatic deletion after retention period

3. **Access Control**
   - Pre-signed URLs for temporary access (valid 1 hour)
   - Server-side encryption (SSE-S3 or SSE-KMS)
   - Access logging enabled

**API Operations:**
```python
# Upload document
upload_document(file, document_id, metadata) -> s3_url

# Download document
download_document(document_id) -> file_stream

# Delete document
delete_document(document_id) -> success

# Generate temporary access URL
get_presigned_url(document_id, expiry=3600) -> url
```

### 4.12 Database

**Purpose:** Store document metadata, processing results, and user data.

**Technology:** PostgreSQL 14 for relational data, MongoDB for unstructured results

**Schema Design:**

**PostgreSQL Tables:**

```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(50) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    document_type VARCHAR(50), -- docx, chat, pdf
    file_size BIGINT,
    storage_path VARCHAR(500),
    confidentiality_level VARCHAR(50),
    uploaded_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP,
    status VARCHAR(50), -- uploaded, processing, completed, failed
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_uploaded_at (uploaded_at)
);

-- Processing jobs table
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    processor VARCHAR(50), -- parser, ner, llm
    status VARCHAR(50), -- queued, processing, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    INDEX idx_document_id (document_id),
    INDEX idx_status (status)
);

-- Extraction results table
CREATE TABLE extraction_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    entity_name VARCHAR(100),
    entity_value TEXT,
    confidence_score DECIMAL(3,2),
    extracted_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_document_id (document_id)
);

-- Audit logs table
CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(50),
    action VARCHAR(100),
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    request_id UUID,
    INDEX idx_user_id (user_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_resource (resource_type, resource_id)
);

-- Users table
CREATE TABLE users (
    id VARCHAR(50) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    department VARCHAR(100),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    INDEX idx_email (email)
);
```

**MongoDB Collections:**

```javascript
// processed_documents collection
{
  "_id": "doc-12345",
  "document_id": "doc-12345",
  "extracted_entities": {
    "Counterparty": "BANK ABC",
    "Notional": "200 mio",
    "ISIN": "FR001400QV82",
    ...
  },
  "raw_output": { /* full processor output */ },
  "processing_metadata": {
    "processor": "ner_service",
    "model_version": "1.1",
    "processing_time_ms": 180,
    "confidence_scores": { ... }
  },
  "created_at": ISODate("2025-01-07T10:30:00Z")
}

// model_metadata collection
{
  "_id": "model-ner-v1.1",
  "model_name": "ner_financial",
  "version": "1.1",
  "training_date": ISODate("2025-01-01T00:00:00Z"),
  "training_examples": 500,
  "accuracy_metrics": {
    "precision": 0.92,
    "recall": 0.89,
    "f1_score": 0.90
  },
  "deployed_at": ISODate("2025-01-02T00:00:00Z"),
  "status": "production"
}
```

**Database Connection Pooling:**
- Min connections: 5
- Max connections: 50
- Connection timeout: 30 seconds
- Idle timeout: 10 minutes

**Backup Strategy:**
- Automated daily backups at 2 AM UTC
- Point-in-time recovery enabled
- Backup retention: 30 days
- Cross-region replication for disaster recovery

### 4.13 Cache Layer

**Purpose:** Improve performance by caching frequently accessed data.

**Technology:** Redis 7.0

**Cached Data:**

1. **Session Data**
   - User sessions and tokens
   - TTL: 1 hour

2. **Document Metadata**
   - Recently accessed documents
   - TTL: 15 minutes

3. **Processing Results**
   - Extraction results for repeat queries
   - TTL: 24 hours

4. **API Rate Limiting**
   - Request counts per user/IP
   - TTL: 1 minute (sliding window)

**Cache Keys Pattern:**
```
session:{user_id} -> session data
document:{document_id}:metadata -> document info
document:{document_id}:results -> extraction results
ratelimit:{user_id}:minute -> request count
```

**Cache Invalidation:**
- When document is updated or deleted
- When processing completes (update results)
- Manual cache clear via admin API

### 4.14 Logging Service

**Purpose:** Centralized logging for debugging, auditing, and compliance.

**Technology:** ELK Stack (Elasticsearch, Logstash, Kibana) or AWS CloudWatch

**Log Levels:**
- DEBUG: Detailed debugging information
- INFO: General informational messages
- WARNING: Warning messages for potential issues
- ERROR: Error messages for failures
- CRITICAL: Critical issues requiring immediate attention

**Log Categories:**

1. **Application Logs**
   - Service startup/shutdown
   - Request/response logs
   - Processing events
   - Error traces

2. **Audit Logs**
   - User authentication/authorization
   - Document access (view, download, delete)
   - Permission changes
   - Configuration changes

3. **Performance Logs**
   - Request latency
   - Processing times
   - Database query performance
   - External API call times

4. **Security Logs**
   - Failed login attempts
   - Unauthorized access attempts
   - Suspicious activity patterns
   - Token validation failures

**Log Format (JSON):**
```json
{
  "timestamp": "2025-01-07T10:30:00.123Z",
  "level": "INFO",
  "service": "document_service",
  "host": "ador-api-01",
  "request_id": "req-abc123",
  "user_id": "user123",
  "action": "document_upload",
  "document_id": "doc-12345",
  "message": "Document uploaded successfully",
  "metadata": {
    "filename": "contract.docx",
    "file_size": 2048000,
    "processing_time_ms": 150
  }
}
```

**Log Retention:**
- Application logs: 30 days
- Audit logs: 7 years (compliance requirement)
- Performance logs: 90 days

**Log Analysis:**
- Real-time dashboards in Kibana
- Automated alerts for errors/warnings
- Daily log summary reports

### 4.15 Monitoring Service

**Purpose:** Monitor system health, performance, and availability.

**Technology:** Prometheus + Grafana or AWS CloudWatch

**Monitored Metrics:**

1. **System Metrics**
   - CPU usage per service
   - Memory usage per service
   - Disk I/O
   - Network throughput

2. **Application Metrics**
   - Request rate (requests per second)
   - Error rate (errors per minute)
   - Response time (p50, p95, p99 percentiles)
   - Queue depth (messages waiting)
   - Active connections

3. **Business Metrics**
   - Documents processed per hour
   - Success rate by document type
   - Average processing time
   - User activity (active users per hour)

4. **External Dependencies**
   - Database connection pool status
   - Storage service availability
   - LLM API response times
   - Cache hit rate

**Alerting Rules:**

```yaml
alerts:
  - name: high_error_rate
    condition: error_rate > 5%
    duration: 5m
    severity: critical
    notification: pagerduty, email

  - name: slow_response_time
    condition: response_time_p95 > 5s
    duration: 10m
    severity: warning
    notification: slack

  - name: queue_backed_up
    condition: queue_depth > 1000
    duration: 15m
    severity: warning
    notification: slack

  - name: service_down
    condition: service_health == down
    duration: 1m
    severity: critical
    notification: pagerduty, sms
```

**Monitoring Dashboard:**
- System overview (all services status)
- Processing pipeline metrics
- Error tracking and analysis
- User activity and usage patterns

### 4.16 Web UI

**Purpose:** User interface for document upload and interaction.

**Technology:** React.js with Material-UI

**Features:**

1. **Document Upload**
   - Drag-and-drop file upload
   - Multiple file selection
   - Progress bar during upload
   - File type validation

2. **Document Management**
   - List all uploaded documents
   - Filter by type, date, status
   - Search by filename or content
   - Batch operations (delete, export)

3. **Processing Status**
   - Real-time status updates via WebSocket
   - Notification when processing completes
   - Error messages with actionable guidance

4. **Results Viewing**
   - Display extracted entities in tables
   - Download results as JSON or CSV
   - Compare results from multiple documents

5. **Settings**
   - User profile management
   - Processing preferences
   - Notification settings

**UI Architecture:**
```
React Components
    |
    v
Redux State Management
    |
    v
API Client (Axios)
    |
    v
REST API / WebSocket
```

---

## 5. Data Flow

### 5.1 Synchronous Processing Flow (Real-time)

Used for small documents requiring immediate results (< 10 seconds processing time).

```
Step 1: User uploads document via Web UI or API
    |
    v
Step 2: API Gateway validates request and authenticates user
    |
    v
Step 3: Document Service receives file
    |
    v
Step 4: ACL Service checks user permissions
    |
    v
Step 5: Storage Service stores document in S3
    |
    v
Step 6: Classifier determines document type
    |
    v
Step 7: Document Service calls appropriate processor directly
    |
    v
Step 8: Processor extracts entities and returns results
    |
    v
Step 9: Results stored in database and cache
    |
    v
Step 10: Response returned to user immediately
```

**Example API Request/Response:**

Request:
```http
POST /api/v1/documents/process-sync
Authorization: Bearer eyJhbGc...
Content-Type: multipart/form-data

file: chat_message.txt
```

Response (after 2 seconds):
```json
{
  "document_id": "doc-12345",
  "status": "completed",
  "processing_time_ms": 1850,
  "entities": {
    "Counterparty": "BANK ABC",
    "Notional": "200 mio",
    "ISIN": "FR001400QV82",
    "Maturity": "2Y EVG",
    "Bid": "estr+45bps",
    "Underlying": "AVMAFC FLOAT 06/30/28",
    "PaymentFrequency": "Quarterly"
  }
}
```

### 5.2 Asynchronous Processing Flow (Batch)

Used for large documents or batch processing (processing time > 10 seconds).

```
Step 1: User uploads document(s) via API
    |
    v
Step 2: API Gateway authenticates and routes request
    |
    v
Step 3: Document Service stores document and creates job
    |
    v
Step 4: Job pushed to Processing Queue
    |
    v
Step 5: Document Service returns job_id immediately
    |
    v
Step 6: Worker picks up job from queue
    |
    v
Step 7: Classifier determines document type
    |
    v
Step 8: Appropriate processor handles document
    |
    v
Step 9: Results stored in database
    |
    v
Step 10: Notification sent to user (email/webhook/WebSocket)
```

**Example API Request/Response:**

Request:
```http
POST /api/v1/documents/process-async
Authorization: Bearer eyJhbGc...
Content-Type: multipart/form-data

file: large_report.pdf
```

Immediate Response:
```json
{
  "job_id": "job-67890",
  "document_id": "doc-12345",
  "status": "queued",
  "estimated_time_seconds": 120,
  "status_url": "/api/v1/jobs/job-67890/status"
}
```

Status Check (polling or WebSocket):
```http
GET /api/v1/jobs/job-67890/status
```

Response when completed:
```json
{
  "job_id": "job-67890",
  "document_id": "doc-12345",
  "status": "completed",
  "processing_time_seconds": 95,
  "results_url": "/api/v1/documents/doc-12345/results"
}
```

### 5.3 Error Flow

When processing fails at any step:

```
Step 1: Processor encounters error
    |
    v
Step 2: Error logged with full context
    |
    v
Step 3: Retry logic checks retry count
    |
    v
Step 4a: If retry_count < max_retries
    |       -> Increment retry_count
    |       -> Re-queue job with exponential backoff
    |       -> Return to processing
    |
    v
Step 4b: If retry_count >= max_retries
    |       -> Mark job as failed
    |       -> Send failure notification to user
    |       -> Move to dead letter queue
    |       -> Alert operations team
```

**Exponential Backoff:**
- Retry 1: Wait 5 seconds
- Retry 2: Wait 15 seconds
- Retry 3: Wait 45 seconds
- After 3 retries: Mark as failed

---

## 6. API Design

### 6.1 API Versioning

All API endpoints include version in URL: `/api/v1/...`

Version changes:
- v1: Initial release
- v2: Breaking changes (when needed, v1 supported for 6 months)

### 6.2 Authentication

All requests require authentication via JWT token in header:
```
Authorization: Bearer <token>
```

### 6.3 Core API Endpoints

#### 6.3.1 Document Management

**Upload Document**
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>

Parameters:
- file: Document file (required)
- confidentiality: string (optional, default: "internal")
- metadata: JSON object (optional)

Response 201:
{
  "document_id": "doc-12345",
  "status": "uploaded",
  "storage_url": "s3://ador-documents/raw/2025/01/07/doc-12345.docx"
}
```

**Get Document Details**
```http
GET /api/v1/documents/{document_id}
Authorization: Bearer <token>

Response 200:
{
  "document_id": "doc-12345",
  "filename": "contract.docx",
  "document_type": "docx",
  "file_size": 2048000,
  "confidentiality": "internal",
  "uploaded_at": "2025-01-07T10:30:00Z",
  "uploaded_by": "user123",
  "status": "completed",
  "processed_at": "2025-01-07T10:30:15Z"
}
```

**List Documents**
```http
GET /api/v1/documents?page=1&limit=20&type=docx&status=completed
Authorization: Bearer <token>

Response 200:
{
  "total": 150,
  "page": 1,
  "limit": 20,
  "documents": [
    {
      "document_id": "doc-12345",
      "filename": "contract.docx",
      "status": "completed",
      "uploaded_at": "2025-01-07T10:30:00Z"
    },
    ...
  ]
}
```

**Delete Document**
```http
DELETE /api/v1/documents/{document_id}
Authorization: Bearer <token>

Response 204: No Content
```

#### 6.3.2 Processing

**Process Document (Synchronous)**
```http
POST /api/v1/documents/{document_id}/process-sync
Authorization: Bearer <token>

Parameters:
- features: array of strings (optional, default: ["ner"])
  Options: ["ner", "summarize", "topics", "qa"]

Response 200:
{
  "document_id": "doc-12345",
  "processing_time_ms": 1850,
  "results": {
    "ner": {
      "entities": { ... }
    }
  }
}
```

**Process Document (Asynchronous)**
```http
POST /api/v1/documents/{document_id}/process-async
Authorization: Bearer <token>

Parameters:
- features: array of strings
- priority: string (optional, default: "normal")
  Options: ["high", "normal", "low"]

Response 202:
{
  "job_id": "job-67890",
  "status": "queued",
  "estimated_time_seconds": 120
}
```

**Get Processing Status**
```http
GET /api/v1/jobs/{job_id}/status
Authorization: Bearer <token>

Response 200:
{
  "job_id": "job-67890",
  "document_id": "doc-12345",
  "status": "processing",
  "progress_percent": 45,
  "started_at": "2025-01-07T10:30:00Z",
  "estimated_completion": "2025-01-07T10:32:00Z"
}
```

#### 6.3.3 Results

**Get Extraction Results**
```http
GET /api/v1/documents/{document_id}/results
Authorization: Bearer <token>

Response 200:
{
  "document_id": "doc-12345",
  "extracted_entities": {
    "Counterparty": "BANK ABC",
    "Notional": "200 mio",
    ...
  },
  "confidence_scores": {
    "Counterparty": 0.95,
    "Notional": 0.92,
    ...
  },
  "processing_metadata": {
    "processor": "ner_service",
    "model_version": "1.1",
    "processed_at": "2025-01-07T10:30:15Z"
  }
}
```

**Export Results**
```http
GET /api/v1/documents/{document_id}/results/export?format=json
Authorization: Bearer <token>

Parameters:
- format: string (required)
  Options: ["json", "csv", "excel"]

Response 200:
Content-Type: application/json (or text/csv, application/vnd.ms-excel)
Content-Disposition: attachment; filename="doc-12345-results.json"

[File download]
```

#### 6.3.4 Health and Status

**Health Check**
```http
GET /health

Response 200:
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "timestamp": "2025-01-07T10:30:00Z"
}
```

**System Status**
```http
GET /api/v1/status
Authorization: Bearer <token>

Response 200:
{
  "api_gateway": "healthy",
  "document_service": "healthy",
  "parser_service": "healthy",
  "ner_service": "healthy",
  "llm_service": "degraded",
  "database": "healthy",
  "storage": "healthy",
  "queue": {
    "status": "healthy",
    "depth": 45,
    "processing_rate": 120
  }
}
```

### 6.4 WebSocket API

For real-time status updates:

```javascript
// Connect
const ws = new WebSocket('wss://api.ador.cmi.com/ws?token=<jwt_token>');

// Subscribe to document updates
ws.send(JSON.stringify({
  "action": "subscribe",
  "document_id": "doc-12345"
}));

// Receive updates
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log(update);
  // {
  //   "document_id": "doc-12345",
  //   "status": "processing",
  //   "progress": 45
  // }
};
```

### 6.5 Error Responses

All errors follow consistent format:

```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "Document with ID doc-12345 not found",
    "details": {
      "document_id": "doc-12345"
    },
    "request_id": "req-abc123",
    "timestamp": "2025-01-07T10:30:00Z"
  }
}
```

**Common Error Codes:**
- 400: BAD_REQUEST
- 401: UNAUTHORIZED
- 403: FORBIDDEN
- 404: DOCUMENT_NOT_FOUND, USER_NOT_FOUND
- 409: DUPLICATE_DOCUMENT
- 413: FILE_TOO_LARGE
- 422: INVALID_FILE_TYPE
- 429: RATE_LIMIT_EXCEEDED
- 500: INTERNAL_SERVER_ERROR
- 503: SERVICE_UNAVAILABLE

---

## 7. Security Architecture

### 7.1 Network Security

**Layered Network Approach:**

```
Internet
    |
    v
[WAF] - Web Application Firewall
    |
    v
[Load Balancer] - SSL/TLS termination
    |
    v
[DMZ Zone] - API Gateway, Web UI
    |
    v
[Application Zone] - Services (no direct internet access)
    |
    v
[Data Zone] - Database, Storage (strict access control)
```

**Firewall Rules:**
- Allow HTTPS (443) from internet to Load Balancer
- Allow HTTP (80) redirect to HTTPS
- Allow services to communicate within private network only
- Block all other inbound traffic
- Allow outbound to specific external services (LLM APIs)

### 7.2 Data Encryption

**Encryption at Rest:**
- Database: Transparent Data Encryption (TDE)
- Storage: AES-256 encryption (S3 SSE or Azure Storage Encryption)
- Backups: Encrypted with separate key

**Encryption in Transit:**
- TLS 1.3 for all external communication
- Internal service communication via mTLS (mutual TLS)
- Certificate rotation every 90 days

**Key Management:**
- AWS KMS or Azure Key Vault for key storage
- Separate keys for different data classifications
- Key rotation policy: Annual for data keys, quarterly for cert keys
- Hardware Security Module (HSM) for sensitive operations

### 7.3 Authentication and Authorization

**Multi-Factor Authentication:**
- Required for all users
- Options: SMS, authenticator app, hardware token
- Grace period: 7 days for new users

**Token Security:**
- JWT tokens with RS256 signature algorithm
- Access token lifetime: 1 hour
- Refresh token lifetime: 7 days
- Tokens stored in httpOnly cookies (Web UI)
- Token revocation list maintained in Redis

**Password Policy:**
- Minimum 12 characters
- Must include uppercase, lowercase, number, special character
- Cannot reuse last 5 passwords
- Expires every 90 days
- Hashed with bcrypt (cost factor 12)

### 7.4 Access Control

**Role Hierarchy:**
```
Admin (full system access)
  |
  +-- Manager (team-level management)
  |     |
  |     +-- Trader (create, read, update documents)
  |     |
  |     +-- Analyst (read, process documents)
  |
  +-- Viewer (read-only access)
```

**Permission Matrix:**

| Role | Upload | Read | Process | Delete | Admin |
|------|--------|------|---------|--------|-------|
| Admin | Yes | All | Yes | All | Yes |
| Manager | Yes | Team | Yes | Team | No |
| Trader | Yes | Own | Yes | Own | No |
| Analyst | No | Shared | Yes | No | No |
| Viewer | No | Shared | No | No | No |

### 7.5 Data Privacy

**Personal Data Handling:**
- Minimal collection (only necessary data)
- Data anonymization for analytics
- User consent tracking
- Right to deletion support
- Data export functionality

**Document Confidentiality:**
- Classification labels (Public, Internal, Confidential, Highly Confidential)
- Watermarking for sensitive documents
- Download restrictions for highly confidential
- Access logging and audit trails

**Compliance:**
- GDPR compliance for EU users
- SOC 2 Type II certified infrastructure
- Regular security audits (quarterly)
- Penetration testing (annual)

### 7.6 Threat Protection

**Input Validation:**
- File type validation (magic number check, not just extension)
- File size limits (100MB max)
- Content scanning for malware (ClamAV)
- XSS protection in API responses
- SQL injection prevention (parameterized queries)
- Command injection prevention (no shell execution)

**Rate Limiting:**
- Per user: 1000 requests per minute
- Per IP: 5000 requests per minute
- Per endpoint: Custom limits (e.g., upload: 10/minute)
- Progressive delays for repeated violations

**DDoS Protection:**
- CloudFlare or AWS Shield
- Automatic traffic filtering
- Geographic restrictions if needed
- Burst capacity for legitimate traffic spikes

**Intrusion Detection:**
- Monitor for suspicious patterns
- Failed authentication attempts tracking
- Unusual data access patterns
- Automated blocking for confirmed threats
- Alerts sent to security team

---

## 8. Document Processing Pipeline

### 8.1 Pipeline Overview

```
Document Upload
    |
    v
Validation (type, size, malware)
    |
    v
Storage (S3/Blob)
    |
    v
Classification (determine type)
    |
    v
Routing (to appropriate processor)
    |
    +---> [DOCX] --> Parser Service
    |                     |
    |                     v
    |              Rule-based extraction
    |                     |
    +---> [Chat] --> NER Service
    |                     |
    |                     v
    |              spaCy + custom patterns
    |                     |
    +---> [PDF] --> LLM Service
                          |
                          v
                   RAG + GPT-4 extraction
                          |
    +---------------------+
    |
    v
Validation (check entity completeness)
    |
    v
Storage (Database + Cache)
    |
    v
Notification (user alert)
```

### 8.2 Processing Stages

**Stage 1: Pre-processing**
- File validation
- Virus scanning
- Format verification
- Size check
- Duplicate detection (hash-based)

**Stage 2: Classification**
- Determine document type
- Calculate confidence score
- Route to appropriate service
- Set processing priority

**Stage 3: Extraction**
- Parse document structure
- Extract entities
- Apply business rules
- Generate confidence scores

**Stage 4: Post-processing**
- Validate extracted data
- Apply data quality checks
- Normalize formats (dates, amounts)
- Calculate aggregate metrics

**Stage 5: Storage**
- Save results to database
- Update cache
- Archive processed document
- Clean up temporary files

**Stage 6: Notification**
- Send completion notification
- Update job status
- Trigger webhooks if configured
- Log completion event

### 8.3 Quality Assurance

**Confidence Scoring:**
- Each extracted entity has confidence score (0.0 - 1.0)
- Low confidence (< 0.7): Flag for manual review
- Medium confidence (0.7 - 0.9): Accept but log
- High confidence (> 0.9): Auto-accept

**Validation Rules:**
- ISIN format validation (2 letters + 10 alphanumeric)
- Date format consistency
- Amount format validation
- Counterparty name validation (against known list)

**Human Review Queue:**
- Low confidence extractions
- Failed validations
- Retry exhausted documents
- User-flagged results

---

## 9. Storage and Database

### 9.1 Data Partitioning

**Document Storage (S3):**
- Partitioned by year/month/day
- Separate buckets for different confidentiality levels
- Cross-region replication for disaster recovery

**Database (PostgreSQL):**
- Partitioned by upload date (monthly partitions)
- Separate tablespaces for active and archived data
- Read replicas for reporting queries

**Results Storage (MongoDB):**
- Sharded by document_id
- Separate collections by document type
- TTL indexes for auto-deletion of old data

### 9.2 Backup and Recovery

**Backup Schedule:**
- Full backup: Daily at 2 AM UTC
- Incremental backup: Every 6 hours
- Transaction logs: Continuous backup
- Cross-region backup copy

**Recovery Time Objective (RTO):** 1 hour
**Recovery Point Objective (RPO):** 15 minutes

**Disaster Recovery Plan:**
1. Detect failure (automated monitoring)
2. Switch to standby region (automated)
3. Restore from latest backup if needed
4. Verify data integrity
5. Resume operations
6. Post-mortem analysis

### 9.3 Data Lifecycle

**Active Data (0-90 days):**
- Hot storage (SSD)
- Frequent access
- Full indexing
- Real-time sync

**Warm Data (90 days - 1 year):**
- Standard storage (HDD)
- Occasional access
- Reduced indexing
- Daily sync

**Cold Data (1-7 years):**
- Archive storage (Glacier)
- Rare access
- Metadata only indexed
- On-demand retrieval (4-12 hours)

---

## 10. Logging and Monitoring

### 10.1 Log Collection

**Log Sources:**
- Application logs from all services
- System logs from servers
- Access logs from API Gateway
- Audit logs from security events
- Performance metrics from monitoring

**Log Aggregation:**
- Logs sent to centralized logging service
- Real-time streaming via Fluentd or Logstash
- Indexed in Elasticsearch
- Searchable via Kibana

### 10.2 Monitoring Dashboards

**Dashboard 1: System Health**
- Service status indicators
- Resource utilization (CPU, memory, disk)
- Network traffic
- Error rates

**Dashboard 2: Processing Pipeline**
- Documents processed (hourly, daily)
- Processing times by document type
- Queue depths
- Success/failure rates

**Dashboard 3: User Activity**
- Active users
- API calls per user
- Most accessed documents
- Feature usage statistics

**Dashboard 4: Business Metrics**
- Total documents processed
- Entity extraction accuracy
- User satisfaction (from feedback)
- Cost per document

### 10.3 Alerting

**Critical Alerts (PagerDuty + SMS):**
- Service completely down
- Database connection lost
- Security breach detected
- Data corruption detected

**Warning Alerts (Slack + Email):**
- High error rate (> 5%)
- Slow response times
- Queue backing up
- Disk space low (< 20%)

**Info Alerts (Slack):**
- Deployment completed
- Scheduled maintenance
- Daily summary reports

---

## 11. Error Handling and Fallback Mechanisms

### 11.1 Retry Logic

**Transient Errors:**
- Network timeouts
- Database connection failures
- External API rate limits

**Retry Strategy:**
```python
def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except TransientError as e:
            wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
            if attempt < max_retries - 1:
                log.warning(f"Attempt {attempt+1} failed, retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                log.error(f"All {max_retries} attempts failed")
                raise
```

### 11.2 Circuit Breaker

Prevents cascading failures when external service is down:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None

    def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise ServiceUnavailableError("Circuit breaker is OPEN")

        try:
            result = func()
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise
```

### 11.3 Fallback Strategies

**Parser Service Failure:**
- Fallback to NER Service for entity extraction
- If both fail, queue for manual processing

**NER Service Failure:**
- Use cached model results if available
- Fall back to regex-only extraction
- If fails, use LLM Service as last resort

**LLM Service Failure:**
- Use local model if available (smaller, less accurate)
- Queue document for later processing
- Notify user of delay

**Database Failure:**
- Switch to read replica for queries
- Queue writes for later
- Use cached data if available
- Alert operations team immediately

**Storage Service Failure:**
- Retry with exponential backoff
- Use backup storage region
- If upload fails, keep document in memory temporarily
- Alert user of temporary issue

### 11.4 Graceful Degradation

When system is under heavy load:

**Level 1: Normal Operation**
- All features available
- Response time < 2 seconds
- No restrictions

**Level 2: Minor Degradation**
- Disable non-essential features (summarization, topic modeling)
- Focus on core NER functionality
- Response time 2-5 seconds
- Queue batch processing only

**Level 3: Moderate Degradation**
- Async processing only (no sync)
- Increased queue wait times
- Disable new user registrations
- Response time 5-10 seconds

**Level 4: Severe Degradation**
- Read-only mode (no new uploads)
- Only serve cached results
- Display maintenance message
- Alert all users of issue

---

## 12. Scalability and Performance

### 12.1 Horizontal Scaling

**Stateless Services:**
All services designed to be stateless for easy scaling:
- Document Service: Scale to 10+ instances
- Parser Service: Scale to 20+ instances
- NER Service: Scale to 15+ instances
- LLM Service: Scale to 5+ instances

**Auto-scaling Rules:**
```yaml
auto_scaling:
  document_service:
    min_instances: 2
    max_instances: 10
    scale_up_threshold:
      cpu_percent: 70
      requests_per_second: 100
    scale_down_threshold:
      cpu_percent: 30
      requests_per_second: 20
```

### 12.2 Database Scaling

**Read Replicas:**
- 3 read replicas for queries
- Primary for writes only
- Automatic failover to replica

**Connection Pooling:**
- PgBouncer for PostgreSQL
- Connection pool size: 50
- Timeout: 30 seconds

**Query Optimization:**
- Indexes on all foreign keys
- Composite indexes for common queries
- Analyze query plans regularly
- Vacuum and reindex monthly

### 12.3 Caching Strategy

**Multi-layer Cache:**

**Layer 1: Application Cache (In-memory)**
- Recently accessed data
- TTL: 5 minutes
- Invalidate on updates

**Layer 2: Redis Cache**
- Document metadata
- Processing results
- Session data
- TTL: 15-60 minutes

**Layer 3: CDN Cache (CloudFront)**
- Static assets (Web UI)
- Document thumbnails
- Public API responses
- TTL: 24 hours

**Cache Invalidation:**
- On document update/delete
- On user permission change
- Manual invalidation via admin API

### 12.4 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Response Time (p95) | < 2 seconds | CloudWatch |
| Document Upload | < 5 seconds | Application logs |
| DOCX Processing | < 10 seconds | Processing queue |
| Chat NER Processing | < 2 seconds | Processing queue |
| PDF LLM Processing | < 60 seconds | Processing queue |
| Database Query Time (p95) | < 100ms | PostgreSQL logs |
| Cache Hit Rate | > 80% | Redis metrics |
| Availability | 99.9% uptime | Monitoring |

---

## 13. Deployment Architecture

### 13.1 Infrastructure

**Cloud Provider:** AWS or Azure

**Regions:**
- Primary: eu-west-1 (Ireland) or West Europe
- Secondary: us-east-1 (Virginia) or East US (for DR)

**Environment Separation:**
- Development: dev.ador-internal.cmi.com
- Staging: staging.ador-internal.cmi.com
- Production: ador.cmi.com

### 13.2 Kubernetes Deployment

**Cluster Setup:**
- 3 master nodes (high availability)
- 10-50 worker nodes (auto-scaled)
- Node size: 8 CPU, 32GB RAM

**Namespaces:**
- ador-api: API Gateway, Document Service
- ador-processing: Parser, NER, LLM services
- ador-data: Database, Redis, Storage proxies
- ador-monitoring: Prometheus, Grafana, logging

**Example Kubernetes Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: document-service
  namespace: ador-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: document-service
  template:
    metadata:
      labels:
        app: document-service
    spec:
      containers:
      - name: document-service
        image: ador/document-service:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: connection-string
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

### 13.3 CI/CD Pipeline

**Pipeline Stages:**

```
1. Code Commit (GitHub/GitLab)
    |
    v
2. Automated Tests
   - Unit tests
   - Integration tests
   - Security scans (Snyk)
    |
    v
3. Build Docker Images
   - Tag with commit SHA
   - Push to container registry
    |
    v
4. Deploy to Dev Environment
   - Automatic deployment
   - Smoke tests
    |
    v
5. Deploy to Staging
   - Manual approval required
   - Full test suite
    |
    v
6. Deploy to Production
   - Manual approval required
   - Blue-green deployment
   - Gradual rollout (10% -> 50% -> 100%)
    |
    v
7. Post-deployment
   - Monitor metrics
   - Automated rollback if errors spike
```

**Deployment Strategy: Blue-Green**
```
Current (Blue)     New (Green)
   |                   |
   v                   v
100% traffic      0% traffic
   |                   |
   +-------------------+
            |
            v
    Gradual shift:
    90% Blue / 10% Green
    50% Blue / 50% Green
    0% Blue / 100% Green
            |
            v
    Green becomes current
    Blue kept for quick rollback
```

### 13.4 Configuration Management

**Environment Variables:**
```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/ador
DATABASE_POOL_SIZE=50

# Storage
S3_BUCKET=ador-documents
S3_REGION=eu-west-1

# External Services
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Security
JWT_SECRET_KEY=...
JWT_EXPIRATION_HOURS=1

# Monitoring
SENTRY_DSN=https://...
LOG_LEVEL=INFO
```

**Secrets Management:**
- AWS Secrets Manager or Azure Key Vault
- Secrets rotated automatically
- Never committed to code
- Injected at runtime

---

## 14. Integration with CMI IS

### 14.1 Integration Points

**1. Authentication Integration**
- ADOR uses CMI Active Directory for authentication
- SAML or OAuth 2.0 integration
- Single Sign-On (SSO) support

**2. Data Integration**
- ADOR can receive documents from CMI trading systems via API
- Push notifications to CMI systems on processing completion
- Shared user database for permissions

**3. Webhook Integration**
```python
# CMI system registers webhook
POST /api/v1/webhooks/register
{
  "url": "https://cmi-system.com/ador-callback",
  "events": ["document.processed", "document.failed"],
  "secret": "shared_secret_for_verification"
}

# ADOR sends notification when document processed
POST https://cmi-system.com/ador-callback
{
  "event": "document.processed",
  "document_id": "doc-12345",
  "timestamp": "2025-01-07T10:30:00Z",
  "results": { ... }
}
```

### 14.2 API Client Libraries

ADOR provides client libraries for easy integration:

**Python Client:**
```python
from ador_client import AdorClient

client = AdorClient(
    api_url="https://ador.cmi.com",
    api_key="your_api_key"
)

# Upload document
result = client.upload_document("contract.docx")
document_id = result["document_id"]

# Process document
entities = client.process_document(document_id, wait=True)
print(entities)
```

**JavaScript Client:**
```javascript
const AdorClient = require('@cmi/ador-client');

const client = new AdorClient({
  apiUrl: 'https://ador.cmi.com',
  apiKey: 'your_api_key'
});

// Upload and process
const result = await client.uploadDocument('contract.docx');
const entities = await client.processDocument(result.documentId);
console.log(entities);
```

### 14.3 Batch Integration

For bulk processing from CMI systems:

```bash
# Upload multiple documents via CLI
ador-cli batch-upload \
  --directory /mnt/shared/documents \
  --pattern "*.docx" \
  --confidentiality internal \
  --wait-for-completion

# Output: CSV with results
document_id,filename,status,entities_json
doc-001,contract1.docx,completed,"{...}"
doc-002,contract2.docx,completed,"{...}"
```

---

## 15. Technology Stack

### 15.1 Backend Services

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| API Framework | FastAPI | 0.104+ | REST API development |
| DOCX Parser | python-docx | 0.8+ | DOCX file parsing |
| NER Model | spaCy | 3.7+ | Named entity recognition |
| LLM Integration | OpenAI API | Latest | GPT-4 for PDF processing |
| Message Queue | RabbitMQ | 3.12+ | Async job processing |
| Task Scheduling | Celery | 5.3+ | Background jobs |
| Vector DB | Pinecone | Latest | RAG embeddings storage |

### 15.2 Data Layer

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Relational DB | PostgreSQL | 14+ | Structured data |
| Document DB | MongoDB | 6.0+ | Unstructured results |
| Cache | Redis | 7.0+ | Session and data cache |
| Object Storage | AWS S3 | Latest | Document storage |
| Search | Elasticsearch | 8.0+ | Full-text search |

### 15.3 Infrastructure

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Orchestration | Kubernetes | 1.28+ | Container management |
| Service Mesh | Istio | 1.20+ | Service communication |
| API Gateway | Kong | 3.4+ | API management |
| Load Balancer | NGINX | 1.24+ | Traffic distribution |
| CDN | CloudFront | Latest | Content delivery |

### 15.4 Monitoring and Logging

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Metrics | Prometheus | 2.48+ | Metrics collection |
| Visualization | Grafana | 10.2+ | Dashboards |
| Logging | ELK Stack | 8.11+ | Log aggregation |
| Tracing | Jaeger | 1.51+ | Distributed tracing |
| Alerts | PagerDuty | Latest | Incident management |

### 15.5 Security

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Authentication | OAuth 2.0 | - | User authentication |
| Authorization | Casbin | 1.51+ | Access control |
| Secrets | AWS Secrets Manager | Latest | Secret management |
| WAF | CloudFlare | Latest | Web firewall |
| Encryption | TLS 1.3 | - | Transport security |

### 15.6 Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Framework | React | 18+ | UI framework |
| UI Library | Material-UI | 5+ | Component library |
| State Management | Redux | 5+ | Application state |
| API Client | Axios | 1.6+ | HTTP client |
| WebSocket | Socket.io | 4+ | Real-time updates |

---

**End of Document**
