#This is for creating a mapping between previously extracted chunk_type
#and the resource_type that we want to use for the final output
#This is a simple mapping that can be easily modified in the future if needed
"""
resource_types.py
==================
Single source of truth for:
  - Valid resource type strings
  - Mapping from parser chunk_type → resource_type
  - Resource ID prefix per type

If you add a new resource type (e.g. "video"), add it here.
No other file needs to change for type registration.
"""

# ------------------------------------------------------------------
# Valid resource types
# These are the only strings that may appear in resource_type.
# Matches the paper's seven categories.
# ------------------------------------------------------------------
print("RESOURCE_TYPES LOADED")
VALID_RESOURCE_TYPES: frozenset[str] = frozenset({
    "explanation",   # explanatory prose, conceptual paragraphs    
    "example",       # worked examples illustrating a concept
    "activity",      # experiments, demonstrations, hands-on tasks
    "exercise",      # assessment questions, intext questions, end exercises
    "diagram",       # figures, labeled illustrations, process diagrams
    "table",         # structured tabular data
    "recall",        # prior knowledge activation, "Can you recall?"
    "scientist_box", # pedagogical boxes about scientists
    "internet_task", # tasks asking students to search the internet
    "mcq",           # multiple choice questions
    "project",       # longer term tasks or practical projects
    "summary",       # chapter or section summaries
    "callout",       # informational callout boxes (Do you know?)
})


# ------------------------------------------------------------------
# Mapping: parser chunk_type → resource_type
# Your existing parser emits chunk_type strings.
# This mapping converts them to benchmark resource_type strings.
# ------------------------------------------------------------------

CHUNK_TYPE_TO_RESOURCE_TYPE: dict[str, str] = {
    "explanation":      "explanation",
    "summary":          "summary",        # now its own distinct type
    "example":          "example",
    "activity":         "activity",
    "exercise":         "exercise",
    "intext_question":  "exercise",       # treated as exercise for benchmark
    "recall":           "recall",
    "scientist_box":    "scientist_box",
    "internet_task":    "internet_task",
    "mcq":              "mcq",
    "project":          "project",
    "callout":          "callout",
}

# Fallback when chunk_type is missing or unrecognised
DEFAULT_RESOURCE_TYPE = "explanation"


def map_chunk_type(chunk_type: str) -> str:
    """
    Convert a parser chunk_type string to a resource_type string.
    Returns DEFAULT_RESOURCE_TYPE if the chunk_type is not recognised.
    """
    return CHUNK_TYPE_TO_RESOURCE_TYPE.get(chunk_type, DEFAULT_RESOURCE_TYPE)


# ------------------------------------------------------------------
# Resource ID prefixes
# ------------------------------------------------------------------

RESOURCE_ID_PREFIX: dict[str, str] = {
    "explanation":   "TXT",    
    "example":       "EXP",
    "activity":      "ACT",
    "exercise":      "EX",
    "diagram":       "IMG",
    "table":         "TBL",
    "recall":        "RCL",
    "scientist_box": "SCI",
    "internet_task": "INT",
    "mcq":           "MCQ",
    "project":       "PRJ",
    "summary":       "SUM",
    "callout":       "CLT",
}


def get_id_prefix(resource_type: str) -> str:
    return RESOURCE_ID_PREFIX.get(resource_type, "TXT")


# ------------------------------------------------------------------
# Which resource types carry text content
# (used to decide which field to populate and how to build index text)
# ------------------------------------------------------------------

TEXT_RESOURCE_TYPES: frozenset[str] = frozenset({
    "explanation", "example", "activity", "exercise", "recall",
    "scientist_box", "internet_task", "mcq", "project", "summary", "callout"
})

NON_TEXT_RESOURCE_TYPES: frozenset[str] = frozenset({
    "diagram", "table"
})


def is_text_resource(resource_type: str) -> bool:
    return resource_type in TEXT_RESOURCE_TYPES


def is_non_text_resource(resource_type: str) -> bool:
    return resource_type in NON_TEXT_RESOURCE_TYPES
