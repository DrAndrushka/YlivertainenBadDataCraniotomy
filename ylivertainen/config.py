#=================================================
#                      IMPORTS
#=================================================
from dataclasses import dataclass

#=================================================
#              Freezing the TaskConfig
#=================================================
@dataclass(frozen=True)
class TaskConfig:
    name: str
    target_column: str
    positive_class: bool | str | int
    task_type: str  # "binary", "multiclass", "regression"
    inclusion_criteria: dict[object]

#=================================================
#                  Define The Task
#=================================================
TIA_MATCH = TaskConfig(
    name="Matching TIA",
    target_column="TIA_match",
    positive_class=True,
    task_type="binary",
    inclusion_criteria={
        'TIA_match': 'non-NaN',
    },
)

ISCHEMIC_STROKE_MATCH = TaskConfig(
    name="Matching Ischemic Stroke",
    target_column="ischemic_match",
    positive_class=True,
    task_type="binary",
    inclusion_criteria={
        'ischemic_match': 'non-NaN',
    },
)

ANY_CEREBROVASCULAR_MATCH = TaskConfig(
    name="Matching Any Cerebrovascular",
    target_column="any_cerebrovascular_match",
    positive_class=True,
    task_type="binary",
    inclusion_criteria={
        'any_cerebrovascular_match': 'non-NaN',
    },
)


#============================================================
#                      HELPER FUNCTIONS
#============================================================

def big_beautiful_print(task: TaskConfig) -> None:
    print("\n" + "🔹" * 30)
    print("🎯 ACTIVE MODELING TASK")
    print("🔹" * 30)
    print(f"🏷️ Name                 → {task.name}")
    print(f"🎯 Target column        → {task.target_column}")
    print(f"✅ Positive class       → {task.positive_class}")
    print(f"🧮 Task type            → {task.task_type}")
    print(f"🫡 Inclusion Criteria   → {task.inclusion_criteria}")
    print("-" * 40)
    print("📢 This TaskConfig is the single source of truth for")
    print("    cohort definition, DDA, EDA, and modeling.\n")


#============================================================
