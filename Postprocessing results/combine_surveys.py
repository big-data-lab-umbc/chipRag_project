import pandas as pd
import numpy as np
import re
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FOLDER = r"C:\Users\User\Downloads\ChipRag_Survey"
OUTPUT_FILE = r"C:\Users\User\Downloads\chipRAG_master.xlsx"

# IMPORTANT:
# These numbers correspond to the 7 rating columns in Qualtrics.
# Change these only if your Qualtrics order was different.
METRIC_MAP = {
    1: "Relevance",
    2: "Actionability",
    3: "HPC Specificity",
    4: "Correctness",
    5: "chip Policy Compliance",
    6: "Hallucination Rate",
    7: "Noise Copying"
}


# ============================================================
# 2. PARSE ONE QUALTRICS FILE
# ============================================================

def parse_qualtrics_file(filepath, survey_version):

    print("\n" + "=" * 65)
    print(f"Reading: {filepath.name}")
    print(f"Survey version: {survey_version}")
    print("=" * 65)

    # --------------------------------------------------------
    # Read Qualtrics export
    #
    # Row 0 = variable names
    # Rows 1 and 2 = Qualtrics metadata/question-description rows
    # --------------------------------------------------------

    df = pd.read_excel(
        filepath,
        header=0,
        skiprows=[1]
    )

    print(f"Rows found before cleaning: {len(df)}")

    # --------------------------------------------------------
    # Remove preview/test submissions
    # --------------------------------------------------------

    if "DistributionChannel" in df.columns:

        channel = (
            df["DistributionChannel"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        preview_count = (channel == "preview").sum()

        if preview_count > 0:
            print(f"Removing {preview_count} preview/test response(s)")

        df = df[channel != "preview"].copy()

    else:
        print("DistributionChannel column not found.")
        print("No automatic preview filtering was performed.")

    # --------------------------------------------------------
    # Keep finished responses when Finished exists
    # --------------------------------------------------------

    if "Finished" in df.columns:

        # Qualtrics may encode Finished as:
        # True/False, 1/0, or strings.
        finished_text = (
            df["Finished"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        valid_finished = finished_text.isin([
            "true",
            "1",
            "yes"
        ])

        if valid_finished.any():

            unfinished_count = (~valid_finished).sum()

            if unfinished_count > 0:
                print(
                    f"Removing {unfinished_count} unfinished "
                    f"response(s)"
                )

            df = df[valid_finished].copy()

    df = df.reset_index(drop=True)

    print(f"Valid submissions after cleaning: {len(df)}")

    # --------------------------------------------------------
    # Create unique submission ID
    #
    # NOTE:
    # This is NOT a unique human participant ID.
    #
    # Example:
    # S1_R1 = first valid submission in Survey 1
    # --------------------------------------------------------

    df["Submission"] = [
        f"{survey_version}_R{i + 1}"
        for i in range(len(df))
    ]

    # --------------------------------------------------------
    # Detect rating columns
    #
    # Expected Qualtrics format:
    #
    # M1Q1_1
    # M1Q1_2
    # ...
    # M9Q25_7
    #
    # M1       = model/system code
    # Q1       = question
    # _1       = metric
    # --------------------------------------------------------

    pattern = re.compile(
        r"^M(\d+)Q(\d+)_(\d+)$",
        re.IGNORECASE
    )

    records = []

    matched_columns = 0

    for col in df.columns:

        col_string = str(col).strip()

        match = pattern.match(col_string)

        if not match:
            continue

        model_number = int(match.group(1))
        question_number = int(match.group(2))
        metric_number = int(match.group(3))

        # Ignore anything outside our 7 rating dimensions
        if metric_number not in METRIC_MAP:
            continue

        matched_columns += 1

        model_code = f"M{model_number}"
        question_id = f"Q{question_number}"
        metric_name = METRIC_MAP[metric_number]

        # ----------------------------------------------
        # Extract each respondent's score
        # ----------------------------------------------

        for _, row in df.iterrows():

            score = pd.to_numeric(
                row[col],
                errors="coerce"
            )

            # Skip missing responses
            if pd.isna(score):
                continue

            records.append({

                "SurveyVersion": survey_version,

                # Submission != unique human evaluator
                "Submission": row["Submission"],

                "Question_ID": question_id,

                "Question_Number": question_number,

                "Model": model_code,

                "Metric": metric_name,

                "Score": float(score)
            })

    print(
        f"Matched rating columns: {matched_columns}"
    )

    print(
        f"Rating records created: {len(records)}"
    )

    if matched_columns == 0:

        print("\nWARNING: No M#Q#_# columns were detected.")
        print("First 30 Excel column names are:")

        for col in list(df.columns)[:30]:
            print("   ", col)

        return pd.DataFrame()

    return pd.DataFrame(records)


# ============================================================
# 3. FIND S1.xlsx ... S9.xlsx
# ============================================================

folder = Path(INPUT_FOLDER)

files = []

for fp in folder.glob("S*.xlsx"):

    # Accept exactly:
    # S1.xlsx
    # S2.xlsx
    # ...
    # S9.xlsx

    match = re.match(
        r"^S(\d+)$",
        fp.stem,
        re.IGNORECASE
    )

    if match:
        files.append(fp)


# Sort numerically instead of alphabetically
files = sorted(
    files,
    key=lambda p: int(
        re.match(
            r"^S(\d+)$",
            p.stem,
            re.IGNORECASE
        ).group(1)
    )
)


print("\nLooking in:")
print(INPUT_FOLDER)

print(
    f"\nFound {len(files)} matching survey files."
)

for fp in files:
    print("   ", fp.name)


# ------------------------------------------------------------
# Check that all 9 are present
# ------------------------------------------------------------

expected = {
    f"S{i}.xlsx"
    for i in range(1, 10)
}

found = {
    fp.name
    for fp in files
}

missing = expected - found

if missing:

    print("\nWARNING: Missing files:")

    for filename in sorted(missing):
        print("   ", filename)

else:

    print("\nAll S1.xlsx through S9.xlsx found.")


# ============================================================
# 4. PROCESS ALL SURVEY FILES
# ============================================================

all_data = []

for fp in files:

    match = re.match(
        r"^S(\d+)$",
        fp.stem,
        re.IGNORECASE
    )

    survey_number = int(match.group(1))

    survey_version = f"S{survey_number}"

    print(
        f"\n→ Processing {fp.name} "
        f"as survey version {survey_version}"
    )

    temp_df = parse_qualtrics_file(
        filepath=fp,
        survey_version=survey_version
    )

    if not temp_df.empty:
        all_data.append(temp_df)


# ------------------------------------------------------------
# Stop if nothing parsed
# ------------------------------------------------------------

if not all_data:

    raise SystemExit(
        "\nERROR: No rating data could be parsed."
    )


# ============================================================
# 5. COMBINE ALL SURVEYS
# ============================================================

master_df = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# 6. ORDER THE DATA
# ============================================================

master_df["Survey_Number"] = (
    master_df["SurveyVersion"]
    .str.extract(r"(\d+)")
    .astype(int)
)

master_df["Model_Number"] = (
    master_df["Model"]
    .str.extract(r"(\d+)")
    .astype(int)
)

master_df = master_df.sort_values(
    by=[
        "Survey_Number",
        "Submission",
        "Question_Number",
        "Model_Number",
        "Metric"
    ]
).reset_index(drop=True)

master_df = master_df[
    [
        "SurveyVersion",
        "Submission",
        "Question_ID",
        "Question_Number",
        "Model",
        "Metric",
        "Score"
    ]
]


# ============================================================
# 7. VALID SUBMISSION COUNTS
# ============================================================

submission_table = (
    master_df[
        [
            "SurveyVersion",
            "Submission"
        ]
    ]
    .drop_duplicates()
)


submission_counts = (
    submission_table
    .groupby("SurveyVersion")
    .size()
    .reset_index(name="Valid_Submissions")
)


print("\n" + "=" * 65)
print("VALID SUBMISSIONS BY SURVEY VERSION")
print("=" * 65)

print(
    submission_counts.to_string(
        index=False
    )
)


total_valid_submissions = len(
    submission_table
)


print("\nTotal valid survey submissions:")
print(total_valid_submissions)


# ============================================================
# 8. SANITY CHECKS
# ============================================================

print("\n" + "=" * 65)
print("DATA CHECK")
print("=" * 65)

print(
    "Survey versions:",
    master_df["SurveyVersion"].nunique()
)

print(
    "Model codes:",
    master_df["Model"].nunique()
)

print(
    "Questions:",
    master_df["Question_ID"].nunique()
)

print(
    "Metrics:",
    master_df["Metric"].nunique()
)

print(
    "Rating records:",
    len(master_df)
)


print("\nModels found:")

print(
    sorted(
        master_df["Model"].unique(),
        key=lambda x: int(
            re.search(r"\d+", x).group()
        )
    )
)


print("\nMetrics found:")

for metric in master_df["Metric"].unique():
    print("   ", metric)


# ============================================================
# 9. WIDE FORMAT
#
# One row =
# one submission × question × model
# ============================================================

wide_df = master_df.pivot_table(

    index=[
        "SurveyVersion",
        "Submission",
        "Question_ID",
        "Question_Number",
        "Model"
    ],

    columns="Metric",

    values="Score",

    aggfunc="first"

).reset_index()


# Remove the pivot-table column label
wide_df.columns.name = None


# Desired column order
metric_order = list(METRIC_MAP.values())

for metric in metric_order:

    if metric not in wide_df.columns:
        wide_df[metric] = np.nan


wide_df = wide_df[
    [
        "SurveyVersion",
        "Submission",
        "Question_ID",
        "Question_Number",
        "Model"
    ]
    +
    metric_order
]


# ============================================================
# 10. DESCRIPTIVE STATISTICS
#
# IMPORTANT:
# This is submission-level descriptive statistics.
#
# Because the same real person may have completed multiple
# survey versions, this is NOT yet participant-balanced.
# ============================================================

descriptive_summary = (

    master_df

    .groupby(
        [
            "Model",
            "Metric"
        ]
    )["Score"]

    .agg(
        Mean="mean",
        SD="std",
        Median="median",
        N="count"
    )

    .round(3)

    .reset_index()
)


# ============================================================
# 11. MODEL × METRIC MEAN TABLE
#
# Easy table for viewing results
# ============================================================

mean_table = (

    master_df

    .pivot_table(
        index="Model",
        columns="Metric",
        values="Score",
        aggfunc="mean"
    )

    .round(3)

    .reset_index()
)

mean_table.columns.name = None


# ============================================================
# 12. PARTICIPANT MAPPING TEMPLATE
#
# You have 5 actual evaluators.
# This sheet lets you later connect each anonymous submission
# to DE / MB / XL / SD / GB.
# ============================================================

participant_mapping = submission_table.copy()

participant_mapping["Participant"] = ""

participant_mapping["Notes"] = ""


# Automatically fill the surveys where we know there is only
# one possible participant from your tracking.

participant_mapping.loc[
    participant_mapping["SurveyVersion"] == "S2",
    "Participant"
] = "XL"

participant_mapping.loc[
    participant_mapping["SurveyVersion"] == "S4",
    "Participant"
] = "XL"


# ============================================================
# 13. SAVE EVERYTHING
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    master_df.to_excel(
        writer,
        sheet_name="Long_Format",
        index=False
    )

    wide_df.to_excel(
        writer,
        sheet_name="Wide_Format",
        index=False
    )

    submission_counts.to_excel(
        writer,
        sheet_name="Submission_Counts",
        index=False
    )

    descriptive_summary.to_excel(
        writer,
        sheet_name="Descriptive_Summary",
        index=False
    )

    mean_table.to_excel(
        writer,
        sheet_name="Mean_Table",
        index=False
    )

    participant_mapping.to_excel(
        writer,
        sheet_name="Participant_Mapping",
        index=False
    )


# ============================================================
# 14. FINISHED
# ============================================================

print("\n" + "=" * 65)

print("DONE")

print("=" * 65)

print(
    f"\nMaster workbook saved to:\n{OUTPUT_FILE}"
)

print("\nSheets created:")

print("  1. Long_Format")
print("  2. Wide_Format")
print("  3. Submission_Counts")
print("  4. Descriptive_Summary")
print("  5. Mean_Table")
print("  6. Participant_Mapping")

print(
    "\nIMPORTANT: 'Submission' identifies a survey submission, "
    "not a unique person."
)