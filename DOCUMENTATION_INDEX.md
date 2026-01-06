# **AURA VERACITY LAB — COMPLETE DOCUMENTATION INDEX**

**Status:** ✅ Design Freeze v1.0 Complete  
**Date:** January 3, 2026  
**Next Phase:** Implementation (12 weeks)

---

## **📚 DOCUMENTATION HIERARCHY**

```
DESIGN FREEZE DELIVERY (THIS IS THE ENTRY POINT)
├── PROJECT_PLAN.md (High-level system overview)
├── ML_SYSTEM_DESIGN.md (Complete multimodal design)
├── MODEL_CONTRACT_v1.md (Locked decisions at a glance)
├── IMPLEMENTATION_ROADMAP.md (Phase-by-phase execution plan)
├── CODE_IMPACT_ANALYSIS.md (Specific code changes needed)
├── DOCUMENTATION_INDEX.md (This file)
└── README.md (Existing project overview)

model-service/
├── README.md (ML service overview)
├── src/
│   ├── models/ (Neural network architectures)
│   ├── data/ (Data loading & preprocessing)
│   ├── preprocess/ (Video/audio extraction)
│   ├── serve/ (Inference APIs)
│   ├── eval/ (Evaluation metrics)
│   ├── train.py (Training loop)
│   └── config/ (Configuration YAML)
└── requirements.txt (Dependencies)

backend/
├── README.md (Backend API overview)
├── IMPLEMENTATION_SUMMARY.md (Current implementation status)
├── FRONTEND_INTEGRATION.md (Frontend-backend integration)
├── FILE_MANIFEST.md (Backend file listing)
└── app/
    ├── main.py (FastAPI app factory)
    ├── routes/ (API endpoints)
    ├── middleware/ (Auth, CORS)
    ├── services/ (Supabase integration)
    └── config/ (Settings)

src/
└── [Frontend React application]
```

---

## **🎯 QUICK NAVIGATION BY ROLE**

### **👨‍💼 Project Manager / Team Lead**

**Start Here:** [DESIGN_FREEZE_DELIVERY.md](DESIGN_FREEZE_DELIVERY.md)  
**Time:** 5 minutes

1. Read [MODEL_CONTRACT_v1.md § One-Page Summary](MODEL_CONTRACT_v1.md#-one-page-summary) (3 min)
2. Check [IMPLEMENTATION_ROADMAP.md § Timeline](IMPLEMENTATION_ROADMAP.md#-timeline) (2 min)
3. Reference [PROJECT_PLAN.md § Execution Phases](PROJECT_PLAN.md#d-execution-phases-in-priority-order) for risk mitigation

**Key Documents:**
- [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) — Daily tracking of tasks
- [MODEL_CONTRACT_v1.md](MODEL_CONTRACT_v1.md) — Locked decisions (don't change)
- [DESIGN_FREEZE_DELIVERY.md](DESIGN_FREEZE_DELIVERY.md) — Sign-off document

---

### **🧠 ML Engineer (Implementation)**

**Start Here:** [CODE_IMPACT_ANALYSIS.md](CODE_IMPACT_ANALYSIS.md)  
**Time:** 30 minutes

1. Read [MODEL_CONTRACT_v1.md § Locked Interface Contracts](MODEL_CONTRACT_v1.md#-locked-interface-contracts) (10 min)
2. Pick Phase 1 task from [IMPLEMENTATION_ROADMAP.md § Phase 1](IMPLEMENTATION_ROADMAP.md#-phase-1-critical-fixes-weeks-1–2) (5 min)
3. Reference [CODE_IMPACT_ANALYSIS.md](CODE_IMPACT_ANALYSIS.md) for specific code changes (10 min)
4. Deep dive: [ML_SYSTEM_DESIGN.md § II–IV](ML_SYSTEM_DESIGN.md) for problem context (5 min)

**Key Documents:**
- [CODE_IMPACT_ANALYSIS.md](CODE_IMPACT_ANALYSIS.md) — Specific code changes per decision
- [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) — Task checklists & acceptance criteria
- [MODEL_CONTRACT_v1.md § Example Inference Flow](MODEL_CONTRACT_v1.md#-example-inference-flow) — Data pipeline visualization
- [ML_SYSTEM_DESIGN.md § XI](ML_SYSTEM_DESIGN.md#xi-model-contract-v1-locked-decisions-) — Architecture rationale

---

### **🏗️ System Architect**

**Start Here:** [PROJECT_PLAN.md](PROJECT_PLAN.md)  
**Time:** 45 minutes

1. Read [PROJECT_PLAN.md § Architecture Breakdown](PROJECT_PLAN.md#b-architecture-breakdown) (15 min)
2. Study [ML_SYSTEM_DESIGN.md § IV](ML_SYSTEM_DESIGN.md#iv-complete-multimodal-pipeline-design) — pipeline diagram (10 min)
3. Review [MODEL_CONTRACT_v1.md § Locked Interface Contracts](MODEL_CONTRACT_v1.md#-locked-interface-contracts) (10 min)
4. Check [IMPLEMENTATION_ROADMAP.md § Development Guidelines](IMPLEMENTATION_ROADMAP.md#-development-guidelines) (10 min)

**Key Documents:**
- [PROJECT_PLAN.md](PROJECT_PLAN.md) — System-level architecture
- [ML_SYSTEM_DESIGN.md](ML_SYSTEM_DESIGN.md) — Detailed ML pipeline
- [MODEL_CONTRACT_v1.md](MODEL_CONTRACT_v1.md) — API contracts & output schemas
- [IMPLEMENTATION_ROADMAP.md § Development Guidelines](IMPLEMENTATION_ROADMAP.md#-development-guidelines) — Code organization

---

### **🧪 QA / Testing Team**

**Start Here:** [IMPLEMENTATION_ROADMAP.md § Success Metrics](IMPLEMENTATION_ROADMAP.md#-success-metrics)  
**Time:** 20 minutes

1. Read [IMPLEMENTATION_ROADMAP.md § Success Metrics](IMPLEMENTATION_ROADMAP.md#-success-metrics) (5 min)
2. Check [IMPLEMENTATION_ROADMAP.md § Testing Strategy](IMPLEMENTATION_ROADMAP.md#-testing-strategy-) (10 min)
3. Reference [MODEL_CONTRACT_v1.md § Output Specification](MODEL_CONTRACT_v1.md#-output-specification) for API validation (5 min)

**Key Documents:**
- [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) — Phase-by-phase acceptance criteria
- [MODEL_CONTRACT_v1.md § Output Specification](MODEL_CONTRACT_v1.md#-output-specification) — Expected JSON format
- [IMPLEMENTATION_ROADMAP.md § Code Review Checklist](IMPLEMENTATION_ROADMAP.md#-code-review-checklist) — Code quality standards

---

### **🚀 DevOps / Deployment**

**Start Here:** [IMPLEMENTATION_ROADMAP.md § Development Guidelines](IMPLEMENTATION_ROADMAP.md#-development-guidelines)  
**Time:** 30 minutes

1. Read [IMPLEMENTATION_ROADMAP.md § Dependencies & Requirements](IMPLEMENTATION_ROADMAP.md#-dependencies--requirements) (10 min)
2. Check [PROJECT_PLAN.md § Risk & Gap Analysis](PROJECT_PLAN.md#e-risk--gap-analysis) for deployment issues (10 min)
3. Reference [IMPLEMENTATION_ROADMAP.md § Blockers & Risks](IMPLEMENTATION_ROADMAP.md#-blockers--risks) (10 min)

**Key Documents:**
- [IMPLEMENTATION_ROADMAP.md § Dependencies & Requirements](IMPLEMENTATION_ROADMAP.md#-dependencies--requirements) — Package management
- [PROJECT_PLAN.md § Risk & Gap Analysis](PROJECT_PLAN.md#e-risk--gap-analysis) — Deployment blockers
- [MODEL_CONTRACT_v1.md § FAQ](MODEL_CONTRACT_v1.md#-faq) — Troubleshooting guide

---

### **📊 Stakeholder / Executive**

**Start Here:** [DESIGN_FREEZE_DELIVERY.md](DESIGN_FREEZE_DELIVERY.md)  
**Time:** 10 minutes

1. Read [DESIGN_FREEZE_DELIVERY.md § Expected Performance](DESIGN_FREEZE_DELIVERY.md#-expected-performance) (3 min)
2. Check [DESIGN_FREEZE_DELIVERY.md § Timeline](DESIGN_FREEZE_DELIVERY.md#-implementation-timeline) (2 min)
3. Review [DESIGN_FREEZE_DELIVERY.md § Sign-Off Checklist](DESIGN_FREEZE_DELIVERY.md#-sign-off-checklist) (5 min)

**Key Documents:**
- [DESIGN_FREEZE_DELIVERY.md](DESIGN_FREEZE_DELIVERY.md) — Complete summary & sign-off
- [MODEL_CONTRACT_v1.md § Performance Targets](MODEL_CONTRACT_v1.md#-performance-targets) — Quantified goals

---

## **📖 DOCUMENT DESCRIPTIONS**

### **1. DESIGN_FREEZE_DELIVERY.md** (5KB)
**Purpose:** Integration document tying all deliverables together  
**Audience:** All teams (entry point)  
**Key Sections:**
- Deliverables summary (4 main documents)
- 14 locked decisions table
- Performance targets (70% → 85%+ AUC)
- Implementation timeline (12 weeks)
- Sign-off checklist

**Read This If:** You're starting fresh or need high-level overview

---

### **2. MODEL_CONTRACT_v1.md** (8KB)
**Purpose:** Concrete specification of all locked architectural decisions  
**Audience:** Everyone (quick reference)  
**Key Sections:**
- One-page summary table
- 🔐 Locked interface contracts (audio, video, fusion, output)
- What changed from current implementation
- Critical configuration (config.yaml template)
- Example inference flow (end-to-end)
- FAQ with common blockers

**Read This If:** You need to understand what changed & why it's locked

---

### **3. ML_SYSTEM_DESIGN.md** (30KB)
**Purpose:** Complete multimodal deepfake detection system specification  
**Audience:** ML Engineers, Architects  
**Key Sections:**
- Forensics problem decomposition (visual + audio artifacts)
- Current state audit (18 gaps, 5 critical)
- Complete pipeline design (ASCII diagram)
- Assumptions vs verified facts
- Missing components (categorized by severity)
- ✅ Section XI: Model Contract v1.0 (LOCKED DECISIONS)
- Phase-by-phase roadmap with AUC targets

**Read This If:** You need deep understanding of the multimodal system design

---

### **4. IMPLEMENTATION_ROADMAP.md** (25KB)
**Purpose:** Phase-by-phase implementation plan with task checklists  
**Audience:** Developers, Project Managers  
**Key Sections:**
- Phase 1: 5 critical fixes (2 weeks)
- Phase 2: 5 high-impact improvements (3 weeks)
- Phase 3: 5 advanced methods (4 weeks)
- Each task has: description, effort estimate, expected AUC gain, detailed subtasks
- Success metrics (quantified targets)
- Development guidelines & code organization
- Testing strategy & code review checklist
- Known risks & mitigations

**Read This If:** You're implementing one of the 15 tasks

---

### **5. CODE_IMPACT_ANALYSIS.md** (12KB)
**Purpose:** Translate locked decisions into specific code changes  
**Audience:** ML Engineers implementing Phase 1–3  
**Key Sections:**
- 5 detailed decision → code change mappings
- Code snippets for each major change
- Testing checklists for verification
- Implementation checklist by phase
- Quick start guide for each developer task

**Read This If:** You're writing code and need specific implementation details

---

### **6. PROJECT_PLAN.md** (40KB, from previous session)**
**Purpose:** High-level system architecture & component responsibilities  
**Audience:** Architects, System Designers  
**Key Sections:**
- High-level overview (constraints, assumptions)
- Architecture breakdown (data flows, per-component responsibilities)
- Component-level responsibilities (per folder)
- Execution phases (Phases 1–4)
- Risk & gap analysis (10 critical issues)
- Developer onboarding guide (30/60/120 minute ramp-up)
- Open questions (flagged for decision)

**Read This If:** You need system-wide understanding & component mapping

---

## **🔗 CROSS-DOCUMENT REFERENCE MAP**

| Question | Documents to Check | Time |
|----------|-------------------|------|
| **What changed from current system?** | [MODEL_CONTRACT_v1.md § What Changed](MODEL_CONTRACT_v1.md#-what-changed-from-current-implementation) | 5 min |
| **What's the timeline?** | [DESIGN_FREEZE_DELIVERY.md § Timeline](DESIGN_FREEZE_DELIVERY.md#-implementation-timeline) | 2 min |
| **How do I implement [task]?** | [CODE_IMPACT_ANALYSIS.md](CODE_IMPACT_ANALYSIS.md) + [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) | 20 min |
| **What's the AUC target?** | [MODEL_CONTRACT_v1.md § Performance Targets](MODEL_CONTRACT_v1.md#-performance-targets) | 3 min |
| **Why was [decision] locked?** | [ML_SYSTEM_DESIGN.md § XI](ML_SYSTEM_DESIGN.md#xi-model-contract-v1-locked-decisions-) | 10 min |
| **What's the data pipeline?** | [MODEL_CONTRACT_v1.md § Interface Contracts](MODEL_CONTRACT_v1.md#-locked-interface-contracts) | 15 min |
| **What are the risks?** | [IMPLEMENTATION_ROADMAP.md § Blockers & Risks](IMPLEMENTATION_ROADMAP.md#-blockers--risks) | 10 min |
| **Where does code go?** | [IMPLEMENTATION_ROADMAP.md § Development Guidelines](IMPLEMENTATION_ROADMAP.md#-development-guidelines) | 5 min |
| **What tests do I write?** | [IMPLEMENTATION_ROADMAP.md § Testing Strategy](IMPLEMENTATION_ROADMAP.md#-testing-strategy-) | 10 min |
| **What's the output format?** | [MODEL_CONTRACT_v1.md § Output Specification](MODEL_CONTRACT_v1.md#-output-specification) | 5 min |

---

## **📋 DOCUMENT CHECKLIST FOR HANDOFF**

**Before starting implementation, ensure you have:**

- [ ] Read [DESIGN_FREEZE_DELIVERY.md](DESIGN_FREEZE_DELIVERY.md) (high-level overview)
- [ ] Understood [MODEL_CONTRACT_v1.md § Locked Decisions](MODEL_CONTRACT_v1.md#-locked-decisions) (what's fixed)
- [ ] Reviewed your assigned phase in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) (your tasks)
- [ ] Checked [CODE_IMPACT_ANALYSIS.md](CODE_IMPACT_ANALYSIS.md) for code specifics (implementation details)
- [ ] Familiarized yourself with [ML_SYSTEM_DESIGN.md § XI](ML_SYSTEM_DESIGN.md#xi-model-contract-v1-locked-decisions-) (why decisions matter)
- [ ] Bookmarked [IMPLEMENTATION_ROADMAP.md § Success Metrics](IMPLEMENTATION_ROADMAP.md#-success-metrics) (acceptance criteria)
- [ ] Saved [MODEL_CONTRACT_v1.md § Output Specification](MODEL_CONTRACT_v1.md#-output-specification) for API validation

---

## **🔐 CRITICAL CONSTRAINTS (READ THIS FIRST)**

**These decisions are LOCKED and cannot be changed without Principal Engineer approval:**

1. **Audio Encoder:** wav2vec2-base (pretrained, speaker-agnostic) — non-negotiable for +5–10% AUC gain
2. **Fusion Strategy:** Cross-modal attention (mid-fusion) — core forensic signal, non-negotiable
3. **Inference Mode:** Asynchronous job queue — non-negotiable for UX (30–60s latency)
4. **Training Dataset:** FaceForensics++ + Celeb-DF — locked (affects learned representations)
5. **Temporal Window:** 1 second (5–10 frames) — locked (architecture dependent)

**Flexible decisions (can change with justification):**

1. **Frame Rate:** 5–10 FPS (tradeoff between latency & granularity)
2. **Video Resolution:** 224×224 (can accept 240p–1080p inputs)
3. **Confidence Calibration:** Temperature scaling (alternative methods possible)
4. **Model Update Cadence:** Offline + manual (A/B testing optional in Phase 3)

---

## **📞 DOCUMENT OWNERSHIP & UPDATES**

| Document | Owner | Last Updated | Next Review |
|----------|-------|--------------|-------------|
| DESIGN_FREEZE_DELIVERY.md | Principal ML Engineer | Jan 3, 2026 | Phase 1 complete (Jan 20) |
| MODEL_CONTRACT_v1.md | Principal ML Engineer | Jan 3, 2026 | Phase 1 complete (Jan 20) |
| ML_SYSTEM_DESIGN.md | Principal ML Engineer | Jan 3, 2026 | Final (Mar 17) |
| IMPLEMENTATION_ROADMAP.md | ML Implementation Lead | Jan 3, 2026 | Weekly (Fridays) |
| CODE_IMPACT_ANALYSIS.md | ML Implementation Lead | Jan 3, 2026 | As tasks complete |
| PROJECT_PLAN.md | Staff-Plus Architect | [Previous session] | As-needed |

**Update Protocol:**
- Design documents (freeze docs) → locked, changes require principal approval
- Implementation roadmap → updated weekly with progress
- Code analysis → updated as tasks complete

---

## **✅ DOCUMENT VERIFICATION CHECKLIST**

**All documents are:**
- ✅ Complete (no sections marked TODO)
- ✅ Cross-referenced (hyperlinks working)
- ✅ Locked (design decisions finalized)
- ✅ Specific (code snippets provided)
- ✅ Actionable (task checklists included)
- ✅ Validated (all decisions have rationale)

**Ready for:** Implementation Phase 1 (Jan 6 start)

---

**Document Version:** 1.0  
**Date:** January 3, 2026  
**Status:** ✅ COMPLETE & LOCKED  
**For Questions:** See [DESIGN_FREEZE_DELIVERY.md § Critical Contacts](DESIGN_FREEZE_DELIVERY.md#-critical-contacts)
