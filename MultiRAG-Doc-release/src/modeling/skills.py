"""Domain modeling skills used by the math-model generator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelingSkill:
    """A lightweight domain skill for optimization model generation."""

    name: str
    description: str
    trigger_keywords: tuple[str, ...]
    retrieval_query_templates: tuple[str, ...]
    candidate_components: tuple[str, ...]
    modeling_guidance: str
    quality_checks: tuple[str, ...] = ()

    def build_queries(self, problem: str) -> list[str]:
        return [template.format(problem=problem) for template in self.retrieval_query_templates]


GENERIC_MODELING_SKILL = ModelingSkill(
    name="generic_optimization",
    description="Generic optimization modeling skill.",
    trigger_keywords=(),
    retrieval_query_templates=(
        "{problem}\n\nmathematical modeling optimization formulation sets parameters "
        "decision variables objective function constraints assumptions",
    ),
    candidate_components=(
        "problem type",
        "sets",
        "parameters",
        "decision variables",
        "objective function",
        "constraints",
        "assumptions",
        "validation notes",
    ),
    modeling_guidance=(
        "Use a conservative generic optimization formulation. Reuse retrieved "
        "notation only when it matches the user problem, and explicitly mark missing data."
    ),
    quality_checks=(
        "Every symbol used in formulas should be defined in sets, parameters, or decision variables.",
        "Do not introduce domain-specific constraints unless the user problem or retrieved evidence supports them.",
        "If a constraint is a generic modeling assumption, state that explicitly in validation.notes.",
    ),
)


HOME_HEALTH_CARE_SKILL = ModelingSkill(
    name="home_health_care_routing_scheduling",
    description=(
        "Home health care routing and scheduling with patient classification, "
        "outsourcing, caregiver skill matching, workload, and time propagation."
    ),
    trigger_keywords=(
        "home health care",
        "hhc",
        "hhcrsp",
        "caregiver",
        "caregivers",
        "patient",
        "patients",
        "vip patient",
        "ordinary patient",
        "outsourcing",
        "居家医疗",
        "家庭医疗",
        "上门护理",
        "护理员",
        "患者",
        "外包",
        "技能等级",
        "工作量",
    ),
    retrieval_query_templates=(
        "{problem}\n\nhome health care routing scheduling optimization model components "
        "patient caregiver visit assignment travel time objective constraints",
        "{problem}\n\nhome health care mathematical formulation objective variables constraints "
        "routing scheduling assignment time windows capacity service duration",
        "{problem}\n\nanalogous optimization formulation for home health care service planning "
        "patient categories outsourcing skill matching workload route construction",
        "{problem}\n\nwhich model components are applicable to this home health care problem "
        "sets parameters decision variables objective constraints assumptions",
    ),
    candidate_components=(
        "patient or service-request sets, only if the problem contains patient/request entities",
        "caregiver/vehicle/resource sets, only if visits are assigned to mobile resources",
        "patient categories such as VIP or ordinary patients, when requested or when retrieved evidence supports priority-aware service design",
        "outsourcing or rejection decisions, when requested or when retrieved evidence supports external-service options",
        "assignment variables linking requests to caregivers/resources, if allocation is part of the decision",
        "routing arc variables, only if route order or travel path decisions are required",
        "arrival, start, completion, or departure time variables, if scheduling or time windows matter",
        "skill-compatibility constraints, when requested or when retrieved evidence supports heterogeneous caregiver qualifications",
        "workload, capacity, or route-duration limits, if resources have limits",
        "synchronized visits, breaks, overtime, multi-center assignment, and preference matching when useful for a richer HHC formulation",
        "cost, travel time, waiting time, tardiness, fairness, or service-level objectives according to the user's goal",
    ),
    modeling_guidance="""For a home health care routing and scheduling problem:
- Treat the retrieved paper as a source of modeling ideas, not as a template.
- First judge which candidate components are applicable to the user's problem.
- Components may be introduced when they are requested by the user, strongly implied by the HHC setting, or repeatedly supported by retrieved model evidence.
- If a component is useful by analogy but not explicitly stated, mark it as an evidence-supported assumption.
- Prefer richer, defensible HHC formulations over overly minimal routing-only models when the user asks for paper-level modeling.
- Use semantic variable names and constraint names for the new problem. Do not reproduce source-paper equation numbers or the full source-paper variable set.
- Keep the formulation defensible: include variables and constraints that serve the user's stated objective, evidence-supported assumptions, or a clear HHC modeling rationale.
""",
    quality_checks=(
        "When optional HHC structures are introduced from evidence, label them as assumptions and adapt their notation.",
        "Do not copy a retrieved paper's full variable set, equation numbering, or complete constraint list.",
        "Use routing arc variables only if route order or travel path decisions are part of the problem.",
        "Use time propagation when arrival times, service durations, time windows, waiting, or route timing matter.",
        "Use skill matching, priority, outsourcing, breaks, overtime, synchronization, or preference matching only when requested or supported by retrieved evidence.",
    ),
)


PRODUCTION_SCHEDULING_SKILL = ModelingSkill(
    name="production_scheduling",
    description=(
        "Production, workshop, job-machine, flow-shop, or job-shop scheduling "
        "with assignment, sequencing, timing, cost, tardiness, and capacity decisions."
    ),
    trigger_keywords=(
        "production scheduling",
        "workshop",
        "shop scheduling",
        "job shop",
        "flow shop",
        "flexible job shop",
        "machine",
        "machines",
        "job",
        "jobs",
        "order",
        "orders",
        "operation",
        "operations",
        "processing time",
        "setup",
        "changeover",
        "makespan",
        "tardiness",
        "due date",
        "车间调度",
        "生产调度",
        "作业车间",
        "柔性作业车间",
        "机器",
        "订单",
        "作业",
        "工序",
        "加工时间",
        "换型",
        "延期",
        "完工时间",
        "加工成本",
    ),
    retrieval_query_templates=(
        "{problem}\n\nproduction scheduling mathematical model jobs machines operations "
        "assignment sequencing start time completion time objective constraints",
        "{problem}\n\njob shop flexible job shop flow shop optimization formulation "
        "processing time machine capacity precedence no-overlap tardiness makespan",
        "{problem}\n\nmachine assignment scheduling cost minimization due date tardiness "
        "setup changeover waiting time delivery objective constraints",
        "{problem}\n\nwhich scheduling model components are applicable sets parameters "
        "decision variables objective function constraints assumptions",
    ),
    candidate_components=(
        "job/order sets, only if the problem contains jobs or production orders",
        "machine/resource sets, only if processing resources are part of the decision",
        "operation-stage sets, only if jobs contain ordered operations or process routes",
        "processing-time and cost parameters, if operation durations or costs are relevant",
        "assignment variables linking jobs/operations to machines, if resources are selectable",
        "sequencing/order variables, only if the order on a machine is a decision",
        "start, completion, waiting, or lateness variables, if timing or due dates matter",
        "makespan, tardiness, processing cost, setup cost, or energy objectives according to the user's goal",
        "precedence constraints, only if jobs have ordered operations",
        "machine capacity or no-overlap constraints, if two jobs cannot process simultaneously on the same machine",
        "setup/changeover constraints, only if sequence-dependent setup is stated or evidenced",
        "delivery, inventory, batching, or outsourcing components, only if the user problem includes them",
    ),
    modeling_guidance="""For a production or workshop scheduling problem:
- Treat retrieved scheduling papers as modeling references, not templates.
- First decide whether the task is pure assignment, single-machine scheduling, parallel-machine scheduling, job shop, flow shop, or flexible job shop.
- Do not add sequencing variables if the problem only asks for assignment.
- If sequencing is needed, link ordering variables to machine assignment so no-overlap constraints activate only for operations on the same machine.
- Avoid exact predecessor/successor equalities unless dummy start/end operations are explicitly defined.
- Include start/completion/tardiness variables only when timing, due dates, or delivery performance matter.
- Omit setup/changeover, batching, transportation, and inventory structures unless the user problem or evidence supports them.
- Use compact semantic constraint names and adapted notation; do not reproduce a source paper's full formulation.
""",
    quality_checks=(
        "If a no-overlap constraint uses big-M, it must be activated by both sequencing and same-machine assignment.",
        "If each operation needs one predecessor/successor, dummy start/end operations or inequality degree constraints are usually required.",
        "If tardiness appears in the objective, due-date and tardiness variables must be defined.",
        "If makespan appears in the objective, a makespan variable and completion-time linking constraints must be defined.",
        "If the user asks only for cost-minimizing assignment, do not introduce sequencing or no-overlap constraints by default.",
    ),
)


AVAILABLE_MODELING_SKILLS = (
    HOME_HEALTH_CARE_SKILL,
    PRODUCTION_SCHEDULING_SKILL,
)


def select_modeling_skill(problem: str) -> ModelingSkill:
    """Pick a domain skill using lightweight keyword evidence."""
    text = (problem or "").lower()
    for skill in AVAILABLE_MODELING_SKILLS:
        hits = sum(1 for keyword in skill.trigger_keywords if keyword.lower() in text)
        if hits >= 1:
            return skill
    return GENERIC_MODELING_SKILL
