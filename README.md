# 🏥 Hospital Resource Management OpenEnv

[![CI](https://github.com/your-org/hospital-openenv/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/hospital-openenv/actions/workflows/ci.yml)
[![OpenEnv 1.0](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://openenv.dev)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-60%20passing-brightgreen)](tests/test_env.py)

A production-grade **OpenEnv** environment simulating real hospital operations. An AI agent manages medication inventory, staff scheduling, patient transfers, and crisis response across a multi-ward hospital network — the same decisions made daily by healthcare administrators.

**Zero external dependencies** beyond numpy for the core environment.

---

## Contents

- [What the agent does](#what-the-agent-does)
- [Quick start](#quick-start)
- [Tasks](#tasks)
- [Observation space](#observation-space)
- [Action space](#action-space)
- [Reward function](#reward-function)
- [Baseline scores](#baseline-scores)
- [HTTP API](#http-api)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Project structure](#project-structure)
- [Contributing](#contributing)

---

## What the agent does

Every simulated day the agent decides:

| Decision | Examples |
|----------|---------|
| **Order medications** | "Order 200 units of vancomycin (standard), rush 50 morphine" |
| **Staff units** | "Assign 4 doctors + 12 nurses to the ICU today" |
| **Transfer patients** | "Move CRITICAL patient to ICU; discharge stable patient from general" |
| **Crisis response** | "Pre-order before supply disruption; triage staffing during staff strike" |

The agent observes the full hospital state: patient census and acuity, medication inventory, pending orders, staff pools, 7-day KPI history, and a noisy 3-day admission forecast.

---

## Quick start

```bash
git clone https://github.com/your-org/hospital-openenv.git
cd hospital-openenv

# No pip install needed — stdlib + numpy only
python -m unittest tests.test_env -v    # 60 tests, all pass
```

```python
from environment import HospitalEnv, AgentAction
from baseline.agent import HeuristicAgent

env   = HospitalEnv()
agent = HeuristicAgent()
obs, cfg = env.reset("task2_multi_ward", seed=42)

while True:
    result = env.step(agent.act(obs, cfg))
    obs = result.observation
    if result.done:
        break

score = env.score()
print(f"Score: {score.score:.4f}  Grade: {score.grade}")
# Score: 0.8070  Grade: B
```

See the [Quickstart Guide](docs/quickstart.md) for full usage examples.

---

## Tasks

| | Task 1 | Task 2 | Task 3 |
|-|--------|--------|--------|
| **Difficulty** | Easy | Medium | Hard |
| **Duration** | 30 days | 60 days | 90 days |
| **Wards** | 1 (General) | 3 (ED + General + ICU) | 4 (ED + General + Surgery + ICU) |
| **Medications** | 4 | 8 | 8 |
| **Arrival rate** | Normal | Normal | +60% surge |
| **Crisis events** | None | None | 5 events |

**Task 3 crisis timeline:**
- Day 15: vancomycin + midazolam supply disruption
- Day 30: staff strike (−40% pool, 6 days)
- Day 45: blood product supply disruption
- Day 60: mass casualty (+30 emergency admissions)

---

## Observation space

```python
obs.day                    # int — current day
obs.patients               # List[Patient] — full census with acuity, meds, risk
obs.bed_allocations        # List[BedAllocation] — occupied/available per unit
obs.med_inventory          # Dict[str, int] — units on hand
obs.pending_med_orders     # List[PendingMedOrder] — deliveries in transit
obs.current_schedule       # List[StaffSchedule] — today's staffing
obs.staff_pools            # List[StaffPool] — pool availability
obs.recent_metrics         # List[DailyMetrics] — last 7 days of KPIs
obs.expected_admissions    # List[dict] — noisy 3-day forecast
obs.cumulative_cost        # float
obs.cumulative_mortality   # int
```

Full schema: [docs/api_reference.md](docs/api_reference.md)

---

## Action space

```python
from environment.models import (
    AgentAction, MedOrderAction, StaffAssignAction,
    BedTransferAction, DischargeAction
)

action = AgentAction(
    med_orders    = [MedOrderAction(med_id="vancomycin", quantity=100, rush=False)],
    staff_assign  = [StaffAssignAction(unit_id="icu", doctors=4, nurses=12, techs=3)],
    bed_transfers = [BedTransferAction(patient_id="a3f9", to_unit_id="icu")],
    discharges    = [DischargeAction(patient_id="7d2e")],
)
```

| Sub-action | Key constraint |
|------------|---------------|
| `med_orders` | Each `med_id` once; `rush=True` → 1-day delivery, 2× cost |
| `staff_assign` | Clamped to pool; reduced during strikes |
| `bed_transfers` | Rejected if destination at capacity |
| `discharges` | LOW / MEDIUM acuity only |

---

## Reward function

Dense reward ∈ [−1.0, +1.0] per step:

```
r = w_svc  × service_level_today
  + w_safe × (1 − mortality_rate × 20)
  − w_cost × (day_cost / revenue)
  + w_bed  × bed_utilisation_score
```

Terminal bonus: **+0.5** if final score ≥ 0.80.

Full rationale: [docs/reward_design.md](docs/reward_design.md)

---

## Baseline scores

Verified — seed 42, single run:

| Task | Random | Heuristic | Greedy | RL target |
|------|--------|-----------|--------|-----------|
| task1_single_ward | 0.948 | **0.969** | 0.955 | 0.98+ |
| task2_multi_ward | 0.681 | **0.807** | 0.688 | 0.85+ |
| task3_surge_crisis | 0.490 | **0.849** | 0.747 | 0.80+ |

```bash
python -m baseline.run_baseline --seeds 3 --json
```

---

## HTTP API

```bash
python server.py   # → http://localhost:7860
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | List tasks |
| `/reset` | POST | Start episode |
| `/step` | POST | Submit action |
| `/state` | GET | Current state |
| `/score` | GET | Grade episode |
| `/demo` | POST | Run full episode server-side |

Full reference: [docs/api_reference.md](docs/api_reference.md)

---

## Deployment

### Local (no deps)
```bash
python server.py
```

### Local (with Gradio dashboard)
```bash
pip install -r requirements.txt && python app.py
```

### Docker
```bash
docker build -t hospital-openenv .
docker run -p 7860:7860 hospital-openenv
```

### Hugging Face Spaces
```bash
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/hospital-openenv
git push hf main
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/quickstart.md](docs/quickstart.md) | Get running in 5 minutes |
| [docs/api_reference.md](docs/api_reference.md) | Full Python + HTTP API |
| [docs/architecture.md](docs/architecture.md) | Internal design and extension guide |
| [docs/reward_design.md](docs/reward_design.md) | Reward function rationale |
| [docs/environment_design.md](docs/environment_design.md) | Clinical modelling details |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, how to add tasks/agents |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [openenv.yaml](openenv.yaml) | OpenEnv 1.0 specification |

---

## Project structure

```
hospital-openenv/
├── openenv.yaml
├── server.py                   stdlib HTTP server
├── app.py                      FastAPI + Gradio (HF Spaces)
├── Dockerfile
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── environment/
│   ├── models.py               typed dataclasses
│   ├── env.py                  HospitalEnv
│   ├── simulator/demand.py     stochastic patient flow
│   ├── tasks/tasks.py          Task 1/2/3
│   └── graders/graders.py      episode graders
├── baseline/
│   ├── agent.py                Random / Heuristic / Greedy
│   ├── run_baseline.py         evaluation script
│   └── expected_scores.json    verified scores
├── tests/test_env.py           60 unittest tests
└── docs/
    ├── quickstart.md
    ├── api_reference.md
    ├── architecture.md
    ├── reward_design.md
    └── environment_design.md
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, how to add tasks and agents, and the PR checklist.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{hospital_openenv_2024,
  title  = {Hospital Resource Management OpenEnv},
  year   = {2024},
  url    = {https://github.com/your-org/hospital-openenv},
  note   = {OpenEnv 1.0 environment for AI agent benchmarking in healthcare operations}
}
```
