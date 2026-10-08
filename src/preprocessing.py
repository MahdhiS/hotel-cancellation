"""
Shared preprocessing for the hotel booking cancellation project.

This module contains only model-independent preparation shared by all models.

Included:
- Preserve booking_id for clustering / later joins
- Remove accidental CSV index columns
- Remove booking-time leakage columns
- Remove invalid zero-adult and zero-night bookings
- Learn missing-value rules from training data only
- Learn rare-category groupings from training data only
- Apply the same learned rules to validation and test data
- Keep an optional country-inclusive output for the later Responsible AI comparison

Not included:
- Encoding
- Scaling
- Outlier capping
- Log transformations
- Class imbalance handling

Those model-specific operations belong in Phase 4 pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype


# ============================================================
# BLOCK 1: PROJECT CONSTANTS
# ============================================================

TARGET_COLUMN = "is_canceled"
ID_COLUMN = "booking_id"


# Columns marked as unavailable at booking time in the
# updated Data Dictionary.
#
# country is handled separately because the project workflow
# requires a later with-country vs without-country comparison.
LEAKAGE_COLUMNS = [
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
    "customer_type",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
]


# Helper columns created during the temporal split workflow.
# They are not part of the final shared classifier feature set.
HELPER_COLUMNS = [
    "arrival_date_month_num",
    "arrival_date",
    "booking_date",
]


# These columns are categorical IDs even when stored numerically.
FORCED_CATEGORICAL_COLUMNS = {
    "agent",
    "company",
    "country",
}


# ============================================================
# BLOCK 2: CLASSIFIER DATA SPLIT HELPER
# ============================================================

def split_classifier_data(
    prepared_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split prepared data into classifier inputs, target and booking IDs.

    Returns
    -------
    X : pd.DataFrame
        Classifier predictor features.

    y : pd.Series
        Target column: is_canceled.

    booking_ids : pd.Series
        Stable booking identifier retained for clustering / joins.

    Notes
    -----
    - booking_id is never used as a classifier predictor.
    - country is excluded from the normal classifier feature matrix.
    """

    required = {ID_COLUMN, TARGET_COLUMN}
    missing = required - set(prepared_df.columns)

    if missing:
        raise ValueError(
            f"Prepared data is missing required columns: {sorted(missing)}"
        )

    booking_ids = prepared_df[ID_COLUMN].copy()
    y = prepared_df[TARGET_COLUMN].copy()

    X = prepared_df.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
            "country",
        ],
        errors="ignore",
    ).copy()

    return X, y, booking_ids


# ============================================================
# BLOCK 3: SHARED PREPROCESSOR
# ============================================================

@dataclass
class SharedPreprocessor:
    """
    Shared data-level preprocessing used by every model.

    Parameters
    ----------
    rare_threshold : float
        Categories representing less than this proportion of the
        training set are grouped into "Other".

    rare_min_unique : int
        Rare grouping is only applied to categorical columns with
        at least this many unique values.

    Important
    ---------
    All data-dependent rules are learned from TRAINING DATA ONLY.
    """

    rare_threshold: float = 0.01
    rare_min_unique: int = 15

    fill_values_: Dict[str, object] = field(
        default_factory=dict,
        init=False,
    )

    frequent_categories_: Dict[str, Set[str]] = field(
        default_factory=dict,
        init=False,
    )

    output_columns_: list[str] = field(
        default_factory=list,
        init=False,
    )

    output_columns_with_country_: list[str] = field(
        default_factory=list,
        init=False,
    )

    fitted_: bool = field(
        default=False,
        init=False,
    )

    # --------------------------------------------------------
    # BLOCK 3A: FIT
    # --------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> "SharedPreprocessor":
        """
        Learn all data-dependent preprocessing rules from train only.
        """

        df = self._base_clean(train_df)
        df = self._remove_invalid_rows(df)

        self.fill_values_ = self._learn_fill_values(df)
        df = self._apply_fill_values(df)

        self.frequent_categories_ = self._learn_rare_categories(df)
        df = self._apply_rare_categories(df)

        # Preserve a country-inclusive column list for the later
        # Responsible AI comparison.
        self.output_columns_with_country_ = list(df.columns)

        # Normal classifier-ready shared data excludes country.
        self.output_columns_ = [
            col
            for col in df.columns
            if col != "country"
        ]

        self.fitted_ = True

        return self

    # --------------------------------------------------------
    # BLOCK 3B: TRANSFORM
    # --------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        *,
        include_country: bool = False,
    ) -> pd.DataFrame:
        """
        Apply train-learned preprocessing rules to any split.

        Parameters
        ----------
        df : pd.DataFrame
            Train, validation or test dataframe.

        include_country : bool
            If True, preserve country for the later Responsible AI
            with-country vs without-country comparison.

        Returns
        -------
        pd.DataFrame
            Shared-prepared dataframe with consistent columns.
        """

        if not self.fitted_:
            raise RuntimeError(
                "Fit SharedPreprocessor on training data first."
            )

        out = self._base_clean(df)
        out = self._remove_invalid_rows(out)
        out = self._apply_fill_values(out)
        out = self._apply_rare_categories(out)

        if include_country:
            expected_columns = self.output_columns_with_country_

        else:
            if "country" in out.columns:
                out = out.drop(columns=["country"])

            expected_columns = self.output_columns_

        missing = [
            col
            for col in expected_columns
            if col not in out.columns
        ]

        if missing:
            raise ValueError(
                f"Missing expected columns: {missing}"
            )

        return (
            out[expected_columns]
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # BLOCK 3C: FIT + TRANSFORM
    # --------------------------------------------------------

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        *,
        include_country: bool = False,
    ) -> pd.DataFrame:
        """
        Fit on training data and immediately transform training data.
        """

        self.fit(train_df)

        return self.transform(
            train_df,
            include_country=include_country,
        )

    # ========================================================
    # BLOCK 4: DETERMINISTIC BASE CLEANING
    # ========================================================

    @staticmethod
    def _base_clean(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply deterministic cleanup that does not learn from the data.

        Handles:
        - booking_id recovery
        - accidental CSV index columns
        - leakage columns
        - temporary helper columns
        - categorical ID typing
        """

        out = df.copy()

        # ----------------------------------------------------
        # 4A: Recover stable booking ID
        # ----------------------------------------------------
        #
        # Current project files contain:
        #   Unnamed: 0   -> original stable row identifier
        #   Unnamed: 0.1 -> accidental CSV export index
        #
        # Preserve the original row identifier as booking_id.

        if ID_COLUMN not in out.columns:

            if "Unnamed: 0" in out.columns:

                out = out.rename(
                    columns={
                        "Unnamed: 0": ID_COLUMN
                    }
                )

            else:
                raise ValueError(
                    "No stable booking ID found. "
                    "'Unnamed: 0' is expected in the current split files."
                )

        # ----------------------------------------------------
        # 4B: Remove remaining accidental index columns
        # ----------------------------------------------------

        extra_index_columns = [
            col
            for col in out.columns
            if col.startswith("Unnamed:")
        ]

        if extra_index_columns:
            out = out.drop(
                columns=extra_index_columns
            )

        # ----------------------------------------------------
        # 4C: Remove booking-time leakage columns
        # ----------------------------------------------------

        leakage_to_drop = [
            col
            for col in LEAKAGE_COLUMNS
            if col in out.columns
        ]

        if leakage_to_drop:
            out = out.drop(
                columns=leakage_to_drop
            )

        # ----------------------------------------------------
        # 4D: Remove workflow-only helper columns
        # ----------------------------------------------------

        helper_to_drop = [
            col
            for col in HELPER_COLUMNS
            if col in out.columns
        ]

        if helper_to_drop:
            out = out.drop(
                columns=helper_to_drop
            )

        # ----------------------------------------------------
        # 4E: Ensure ID-like columns are categorical
        # ----------------------------------------------------

        for col in ["agent", "company"]:

            if col in out.columns:
                out[col] = out[col].astype("string")

        if "country" in out.columns:
            out["country"] = out["country"].astype("string")

        return out

    # ========================================================
    # BLOCK 5: INVALID BOOKING REMOVAL
    # ========================================================

    @staticmethod
    def _remove_invalid_rows(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove invalid booking records.

        Resolves:
        - zero-guest / zero-adult bookings
        - children-only bookings
        - zero-night bookings
        """

        out = df.copy()

        # ----------------------------------------------------
        # 5A: Zero-adult and children-only bookings
        # ----------------------------------------------------
        #
        # adults == 0 covers both:
        # - bookings with no guests
        # - bookings where only children/babies are present

        if "adults" in out.columns:
            out = out[
                out["adults"] > 0
            ]

        # ----------------------------------------------------
        # 5B: Zero-night bookings
        # ----------------------------------------------------

        if "total_nights" in out.columns:

            out = out[
                out["total_nights"] > 0
            ]

        else:

            required = {
                "stays_in_weekend_nights",
                "stays_in_week_nights",
            }

            if required.issubset(out.columns):

                total_nights = (
                    out["stays_in_weekend_nights"]
                    + out["stays_in_week_nights"]
                )

                out = out[
                    total_nights > 0
                ]

        return out.copy()

    # ========================================================
    # BLOCK 6: MISSING-VALUE HANDLING
    # ========================================================

    def _learn_fill_values(
        self,
        train_df: pd.DataFrame,
    ) -> Dict[str, object]:
        """
        Learn missing-value replacements from TRAIN only.

        Numeric columns:
            median

        Categorical columns:
            mode

        The current basic-cleaned project data is expected to have
        no remaining missing values, so this also works as a safety
        mechanism for future splits.
        """

        fill_values = {}

        for col in train_df.columns:

            if col in {
                TARGET_COLUMN,
                ID_COLUMN,
            }:
                continue

            series = train_df[col]

            if (
                is_numeric_dtype(series)
                and col not in FORCED_CATEGORICAL_COLUMNS
            ):

                if series.isna().any():
                    fill_values[col] = series.median()

            else:

                if series.isna().any():

                    mode = series.mode(
                        dropna=True
                    )

                    if not mode.empty:
                        fill_values[col] = mode.iloc[0]

                    else:
                        fill_values[col] = "Missing"

        return fill_values

    def _apply_fill_values(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply missing-value replacements learned from training data.
        """

        out = df.copy()

        for col, value in self.fill_values_.items():

            if col in out.columns:
                out[col] = out[col].fillna(value)

        return out

    # ========================================================
    # BLOCK 7: RARE-CATEGORY GROUPING
    # ========================================================

    def _learn_rare_categories(
        self,
        train_df: pd.DataFrame,
    ) -> Dict[str, Set[str]]:
        """
        Learn frequent categories from TRAIN only.

        Categories below rare_threshold are treated as rare.

        Rare grouping is only applied to high-cardinality
        categorical columns.
        """

        frequent_categories = {}

        for col in train_df.columns:

            if col in {
                TARGET_COLUMN,
                ID_COLUMN,
            }:
                continue

            series = train_df[col]

            # A column is categorical when:
            # - it is explicitly an ID/category column, or
            # - pandas does not consider it numeric.
            categorical = (
                col in FORCED_CATEGORICAL_COLUMNS
                or not is_numeric_dtype(series)
            )

            if not categorical:
                continue

            # Avoid unnecessary grouping for low-cardinality columns.
            if (
                series.nunique(dropna=False)
                < self.rare_min_unique
            ):
                continue

            values = series.astype("string")

            frequencies = values.value_counts(
                normalize=True,
                dropna=False,
            )

            keep = set(
                frequencies[
                    frequencies >= self.rare_threshold
                ]
                .index
                .astype(str)
            )

            frequent_categories[col] = keep

        return frequent_categories

    def _apply_rare_categories(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Replace rare and unseen categories with "Other".
        """

        out = df.copy()

        for col, keep in self.frequent_categories_.items():

            if col not in out.columns:
                continue

            values = out[col].astype("string")

            out[col] = values.where(
                values.isin(keep),
                "Other",
            )

        return out
