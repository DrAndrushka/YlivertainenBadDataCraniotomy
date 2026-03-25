# ============ imports ============
import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import trim_mean, iqr, entropy

#============ ANSI Escape Codes for formating ============
from ylivertainen.aesthetics_helpers import GREEN, YELLOW, ORANGE, RED, BOLD, BLUE, GRAY, RESET

#================================================
#=============== Class definition ===============
#================================================
class YlivertainenDDA:

    #============ Ship's Log: Registering the Loot ============
    def __init__(self, root, post_cohort_df: pd.DataFrame, metadata):
        self.root = root
        self.DDA = post_cohort_df.copy()
        self.task_name = metadata['task_name']
        self.target = metadata['target_col']

        target_to_first = ['row_id', 'target'] + [col for col in self.DDA.columns if col not in ['row_id', 'target']]
        self.DDA = self.DDA[target_to_first]

        self.cols_to_not_analyse = []

        # ========== DDA Tables ==========
        self.numerical_DDA = None
        self.categorical_DDA = None
        self.binary_DDA = None
    
    #======================================================
    #                Drop unneeded columns 
    #======================================================
    def keep_from_analysis(self, cols):
        if len(cols) != 0:
            for col in cols:
                self.cols_to_not_analyse.append(col)
            print(f'KEPT FROM ANALYSIS: {cols}')
            print(f'✅ ALL OTHER COLS WILL BE ANALYSED ✅{RESET}')
        else:
            print(f'{BOLD}Nothing passed in cols')
            print(f'✅ ALL COLS WILL BE ANALYSED ✅{RESET}')
        return self
    
    #======================================================
    #                  Dataset Overview 
    #======================================================
    def dataset_overview(self, id_columns):
        
        print(f'{BOLD}==================================================={RESET}')
        print(f'{BOLD}🧨🧨🧨 ========= Rows & Columns ========= 🧨🧨🧨{RESET}')
        print(f'{BOLD}==================================================={RESET}')
        row_count = len(self.DDA)
        col_count = len(self.DDA.columns)
        print(f'{BOLD}🏗️ Row count:{RESET} {row_count}')
        print(f'{BOLD}🏗️ Columns count:{RESET} {col_count}')

        print('\n' + f'{BOLD}==================================================={RESET}')
        print(f'{BOLD}🧨🧨🧨 ========= Dtype summary ========== 🧨🧨🧨{RESET}')
        print(f'{BOLD}==================================================={RESET}')
        max_len = max(len(col) for col in self.DDA.columns)
        for col in self.DDA.columns:
            col_name = f'{BLUE}{BOLD}{col:<{max_len}}{RESET}'
            col_dtype = self.DDA[col].dtype
            print(f'💈 {col_name} : {col_dtype}')
    
        print('\n' + f'{BOLD}======================================================{RESET}')
        print(f'{BOLD}🧨🧨🧨 ==== Unique vs Repeat encounters ==== 🧨🧨🧨{RESET}')
        print(f'{BOLD}======================================================{RESET}')
        valid_ids = [c for c in id_columns if c in self.DDA.columns]
        missing_ids = [c for c in id_columns if c not in self.DDA.columns]
        if len(id_columns) > 0:
            if missing_ids:
                raise ValueError(f'{BOLD}{RED}❌ ID columns not found:{RESET} {missing_ids}')
            else:
                len_df = len(self.DDA)
                same_id_count = self.DDA.duplicated(subset=valid_ids, keep='first').sum()
                same_id_pct = same_id_count / len_df * 100
                unique_count = len_df - same_id_count
                print(f'{BOLD}☝️ ID columns:{RESET} {BLUE}{BOLD}{", ".join(valid_ids)}{RESET}')
                print(f'{BOLD}☝️ Unique encounters:{RESET} {unique_count} ')
                print(f'{BOLD}☝️ Repeat encounters:{RESET} {same_id_count} ({same_id_pct}%)')
                print('Note: not including the first. Only real value duplicates')
        else:
            print(f'{BOLD}❌ Dataset has no ID columns{RESET}')

    #======================================================
    #               DEEPER Dataset Overview 
    #======================================================

    #================ Numerical Analysis ==================    
    def analyse_numerical(self):
        print('\n' + f'{BOLD}==================================================={RESET}')
        print(f'{BOLD}🔢🔢🔢 ============ NUMERICAL =========== 🔢🔢🔢{RESET}')
        print(f'{BOLD}==================================================={RESET}')

        maintenance_cols = ['included_in_cohort', 'row_id', 'cohort_exclusion_reason', 'target']
        numerical_cols = [col for col in self.DDA.select_dtypes(include="number", exclude="category") if not (self.DDA[col].nunique() == 2) and col not in maintenance_cols and col not in self.cols_to_not_analyse]

        if len(numerical_cols) == 0:
            print(f'❌ There are no numerical columns in this Dataset')
            return self

        # ============= HELPER functions ==============
        def _numerical_analysis_function(col):
            col_to_analyse = self.DDA[col]
            len_df = len(col_to_analyse)
            missing_n = col_to_analyse.isna().sum()
            col_name = col
            #============ THE DATA ============
            col_name = col
            n = col_to_analyse.dropna().count()
            n_unique = col_to_analyse.nunique()
            missing_pct = missing_n / len_df
            maxx = col_to_analyse.max()
            minn = col_to_analyse.min()
            median = col_to_analyse.median()
            mean = col_to_analyse.mean()
            mode = [str(m) for m in col_to_analyse.mode()]
            std = col_to_analyse.std()
            cv = std / mean
            trimmed_mean = trim_mean(col_to_analyse, proportiontocut=0.1)
                #Conservative: 5% (0.05). Use this if your data is mostly clean but has a few glitches
                #Standard: 10% (0.1). The sweet spot
                #Aggressive: 25% (0.25). This leaves you with only the middle 50% (the Interquartile Range). Only do that if the data is absolute trash
            iqrange = iqr(col_to_analyse.to_numpy())
            p_5th  = np.percentile(col_to_analyse.to_numpy(), 5)
            p_95th  = np.percentile(col_to_analyse.to_numpy(), 95)
            skewness = col_to_analyse.skew()
                # Positive - Right-Skewed
                # Negative - Left-Skewed
                # 0 - Perfectly Symmetrical: The "ideal" patient population. Mean = Median
            kurtosis = col_to_analyse.kurt()
                # 3 (or 0)* - Mesokurtic
                # Positive (>0) - Leptokurtic - A sharp, skinny peak with Fat Tails - Many "extreme" outliers (critical values) that can't be ignored
                # Negative (<0) - Platykurtic - A flat, broad peak with thin tails - "spread out" but lacks extreme black-swan events            
            
            return col_name, n, n_unique, missing_pct, maxx, minn, median, mean, mode, std, cv, trimmed_mean, iqrange, p_5th, p_95th, skewness, kurtosis

        def _numerical_print(col_name, n, n_unique, missing_pct, maxx, minn, median, mean, mode, std, cv, trimmed_mean, iqrange, p_5th, p_95th, skewness, kurtosis):
            print(f'\n{BLUE}{BOLD}===== {col_name} ====={RESET}')
            print(f'{BOLD}n:{RESET} {n}')
            print(f'{BOLD}n_unique:{RESET} {n_unique}')
            print(f'{BOLD}Missing %:{RESET} {missing_pct}%')
            print(f'{BOLD}MIN:{RESET} {minn}. {BOLD}MAX:{RESET} {maxx}')
            print(f'{BOLD}Median:{RESET} {median}')
            print(f'{BOLD}Mean:{RESET} {mean:.2f}')
            print(f'{BOLD}Trimmed mean:{RESET} {trimmed_mean:.2f}')
            print(f'{BOLD}Mode:{RESET} {", ".join(mode)}')
            print(f'{BOLD}IQR:{RESET} {iqrange}')
            print(f'{BOLD}5%:{RESET} {p_5th:.2f}. {BOLD}95%:{RESET} {p_95th:.2f}')
            print(f'{BOLD}std:{RESET} {std:.2f}')
            if cv < 0.10:
                print(f"{BOLD}cv:{RESET} {cv:.2f} ➡️ Very Low Variability - extremely stable, almost constant")
            elif cv < 0.30:
                print(f"{BOLD}cv:{RESET} {cv:.2f} ➡️ Low Variability - tight around mean, very consistent")
            elif cv < 0.50:
                print(f"{BOLD}cv:{RESET} {cv:.2f} ➡️ Moderate Variability - normal clinical spread")
            elif cv < 1.00:
                print(f"{BOLD}cv:{RESET} {cv:.2f} ➡️ High Variability - noisy feature, check outliers|transforms")
            else:
                print(f"{BOLD}cv:{RESET} {cv:.2f} ➡️ Extreme Variability - highly chaotic relative to mean; interpret with caution")
            if skewness < -0.1:
                print(f'{BOLD}Skewness:{RESET} {skewness:.2f} ➡️ Left-Skewed. Most data are \"high\"')
            elif skewness > 0.1:
                print(f'{BOLD}Skewness:{RESET} {skewness:.2f} ➡️ Right-Skewed. Most data are \"low\"')
            else:
                print(f'{BOLD}Skewness:{RESET} {skewness:.2f} ➡️ Perfectly Symmetrical. A \"perfect\" Bell Curve')
            if kurtosis < -0.5:
                print(f'{BOLD}Kurtosis:{RESET} {kurtosis:.2f} ➡️ Platykurtic. Data is spread out, extreme outliers are rare')
            elif kurtosis > 0.5:
                print(f'{BOLD}Kurtosis:{RESET} {kurtosis:.2f} ➡️ Leptokurtic. Many extreme "Black Swan" outliers')
            else:
                print(f'{BOLD}Kurtosis:{RESET} {kurtosis:.2f} ➡️ Mesokurtic. Normal peak, normal tails. \"Goldilocks\"')
            
            return

        def _numerical_graphs(col_name):
            n_unique = self.DDA[col_name].nunique()
            if n_unique <= 20:
                plt.figure(figsize=[10,2])

                ax1 = plt.subplot(1,2,1)
                sns.histplot(x=col_name, data=self.DDA, stat='density', discrete=True)
                ax1.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
                plt.xticks(rotation=10, ha="right")

                ax2 = plt.subplot(1,2,2)
                sns.boxplot(x=col_name, data=self.DDA)
                ax2.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
                plt.xticks(rotation=10, ha="right")

                plt.subplots_adjust(wspace=0.2)

                plt.show()
                plt.close()
            else:
                plt.figure(figsize=[10,2])

                ax1 = plt.subplot(1,2,1)
                sns.histplot(x=col_name, data=self.DDA, stat='density', kde=True)
                ax1.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
                plt.xticks(rotation=10, ha="right")

                ax2 = plt.subplot(1,2,2)
                sns.boxplot(x=col_name, data=self.DDA)
                ax2.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
                plt.xticks(rotation=10, ha="right")

                plt.subplots_adjust(wspace=0.2)

                plt.show()
                plt.close()

            return
        # ========== END of HELPER functions ==========

        # === AWESOME BINARY TABLE ===
        numerical_overview_table_data = []
        # === END of TABLE ===

        for col in numerical_cols:
            # Calculations
            col_name, n, n_unique, missing_pct, maxx, minn, median, mean, mode, std, cv, trimmed_mean, iqrange, p_5th, p_95th, skewness, kurtosis = _numerical_analysis_function(col)
            numerical_overview_table_data.append([col_name, n, n_unique, missing_pct, maxx, minn, median, mean, mode, std, cv, trimmed_mean, iqrange, p_5th, p_95th, skewness, kurtosis])
            # Prints
            _numerical_print(col_name, n, n_unique, missing_pct, maxx, minn, median, mean, mode, std, cv, trimmed_mean, iqrange, p_5th, p_95th, skewness, kurtosis)
            # Graphs
            _numerical_graphs(col_name)
        
        numerical_overview_table = pd.DataFrame(
            numerical_overview_table_data,
            columns=[
                'column', 'n', 'n_unique', 'missing_%',
                'max', 'min', 'median', 'mean', 'mode',
                'std', 'cv', 'trimmed_mean', 'iqrange', 'p_5th', 'p_95th',
                'skewness', 'kurtosis'
            ])
        
        display(numerical_overview_table
                .style
                .format({"n": "{:.0f}", "n_unique": "{:.0f}", "missing_%": "{:.2f}", "max": "{:.0f}",
                         "min": "{:.0f}", "median": "{:.1f}", "mean": "{:.3f}",
                         "std": "{:.3f}", "cv": "{:.3f}", "trimmed_mean": "{:.3f}",
                         "iqrange": "{:.2f}", "p_5th": "{:.2f}", "p_95th": "{:.2f}",
                         "skewness": "{:.2f}", "kurtosis": "{:.2f}"})
                .background_gradient(
                    subset=['n', 'n_unique', 'missing_%',
                            'max', 'min', 'median', 'mean',
                            'std', 'cv', 'trimmed_mean', 'iqrange', 'p_5th', 'p_95th',
                            'skewness', 'kurtosis'],
                    cmap="Reds"))

        self.numerical_DDA = numerical_overview_table
        
        return self

    #=============== Categorical Analysis =================
    def analyse_categorical(self):
        print('\n' + f'{BOLD}==================================================={RESET}')
        print(f'{BOLD}🚦🚦🚦 ============= CATEGORICAL ============= 🚦🚦🚦{RESET}')
        print(f'{BOLD}==================================================={RESET}')

        maintenance_cols = ['included_in_cohort', 'row_id', 'cohort_exclusion_reason', 'target']
        categorical_cols = [col for col in self.DDA.select_dtypes(include="category", exclude=['bool', 'boolean']) if not col.endswith("_missing") and col not in maintenance_cols and (len(self.DDA[col].value_counts()) > 2)  and col not in self.cols_to_not_analyse]

        # ============= HELPER functions ==============
        def __categorical_balance(col):
            vc = col.value_counts(normalize=True, dropna=True)
            p = vc.values

            if len(p) <= 1:
                return 0.0

            # ignore zero probabilities to avoid log(0)
            p = p[p > 0]

            entropy = -(p * np.log(p)).sum()
            max_entropy = np.log(len(p))  # use original len, not len(p>0)
            if max_entropy == 0:
                return 0.0

            balance = entropy / max_entropy
            return float(balance)
        
        def __median_category(series: pd.Series):
            if not (pd.api.types.is_categorical_dtype(series) and series.cat.ordered):
                return pd.NA

            vc = series.value_counts(normalize=True).reindex(series.cat.categories, fill_value=0)
            cumsum = vc.cumsum()

            mask = cumsum >= 0.5
            if not mask.any():
                return None  # safety, but shouldn't happen

            # first category whose cumulative prob crosses 0.5
            return cumsum.index[mask][0]

        def _categorical_analysis_function(col):
            col_to_analyse = self.DDA[col]
            vc = col_to_analyse.value_counts(normalize=True, dropna=True)
            len_df = len(col_to_analyse)
            missing_n = col_to_analyse.isna().sum()
            
            #============ THE DATA ============
            col_name = col
            n = col_to_analyse.count()
            n_unique = col_to_analyse.nunique()
            missing_pct = missing_n / len_df * 100
            first_mode = vc.index[0]      # most common category
            second_mode = vc.index[1]
            label = str(vc.index[-1])
            p = float(vc.iloc[-1])
            rarest = f"{label}: {p:.2%}"
            first_mode_pct = vc.iloc[0].item() * 100        # already proportion if normalize=True
            second_mode_pct = vc.iloc[1].item() * 100
            max_class_imbalance = first_mode_pct / second_mode_pct
            if pd.api.types.is_categorical_dtype(col_to_analyse) and col_to_analyse.cat.ordered:
                median_category = __median_category(col_to_analyse)
            else:
                median_category = pd.NA
            balance = __categorical_balance(col_to_analyse)
            probs = vc / vc.sum()
            entropy_bin = float(entropy(probs, base=2))
            
            return col_name, n, n_unique, missing_pct, first_mode, second_mode, rarest, first_mode_pct, second_mode_pct, max_class_imbalance, median_category, balance, entropy_bin
        
        def _categorical_print(col_name, n, n_unique, missing_pct, first_mode, second_mode, rarest, first_mode_pct, second_mode_pct, max_class_imbalance, median_category, balance, entropy_bin):
            print(f'\n{BLUE}{BOLD}===== {col_name} ====={RESET}')
            print(f'{BOLD}n:{RESET} {n}')
            print(f'{BOLD}Unique count:{RESET} {n_unique}')
            if n_unique == 1:
                print(f'{BOLD}{RED}Cardinality: {n_unique} ➡️ Constant feature ➡️ Drop for modeling{RESET}')
            elif 2 <= n_unique <= 5:
                print(f'{BOLD}{GREEN}Cardinality: {n_unique} ➡️ Very low ➡️ Ideal for standard one-hot|tree models{RESET}')
            elif 6 <= n_unique <= 20:
                print(f'{BOLD}{GREEN}Cardinality: {n_unique} ➡️ Moderate ➡️ Usually fine; watch for sparse rare levels{RESET}')
            elif 21 <= n_unique <= 50:
                print(f'{BOLD}{ORANGE}Cardinality: {n_unique} ➡️ High ➡️ Consider grouping rare levels or using target|impact encoding{RESET}')
            elif 51 <= n_unique <= 200:
                print(f'{BOLD}{ORANGE}Cardinality: {n_unique} ➡️ Very high ➡️ Strongly consider grouping or specialized encoders{RESET}')
            else:  # n_unique > 200
                print(f'{BOLD}{RED}Cardinality: {n_unique} ➡️ Extreme (ID-like) ➡️ Likely leak/noise ➡️ Treat as ID or drop{RESET}')
            print(f'{BOLD}missing_pct:{RESET} {missing_pct:.2f}%')
            print(f'{BOLD}First MODE:{RESET} {first_mode}')
            print(f'{BOLD}Proportion of MODE:{RESET} {first_mode_pct:.2f}%')
            print(f'{BOLD}second_mode:{RESET} {second_mode}')
            print(f'{BOLD}second_mode_pct:{RESET} {second_mode_pct:.2f}%')
            print(f'{BOLD}rarest:{RESET} {rarest}')
            print(f'{BOLD}max_class_imbalance:{RESET} {max_class_imbalance:.2f}')
            if max_class_imbalance < 1.5:
                print(f'{BOLD}{GREEN}Imbalance ratio: {max_class_imbalance:.2f} ➡️ Classes are similarly frequent ➡️ No special handling needed{RESET}')
            elif 1.5 <= max_class_imbalance < 3:
                print(f'{BOLD}{GREEN}Imbalance ratio: {max_class_imbalance:.2f} ➡️ Mild dominance of the top class ➡️ Usually fine, just monitor minority performance{RESET}')
            elif 3 <= max_class_imbalance < 10:
                print(f'{BOLD}{ORANGE}Imbalance ratio: {max_class_imbalance:.2f} ➡️ Strong top-class dominance ➡️ Consider class weights / resampling for modeling{RESET}')
            else:  # >= 10
                print(f'{BOLD}{RED}Imbalance ratio: {max_class_imbalance:.2f} ➡️ Extreme imbalance ➡️ Treat as rare-event problem; oversampling / specialized methods likely needed{RESET}')
            print(f'{BOLD}median_category:{RESET} {median_category}')
            if 0 <= entropy_bin < 0.1:
                print(f'{BOLD}{RED}Entropy: {entropy_bin:.2f} ➡️ The Flatline ➡️ Almost every single row is identical ➡️ best action --> Drop{RESET}')
            elif 0.1 <= entropy_bin < 1.5:
                print(f'{BOLD}{ORANGE}Entropy: {entropy_bin:.2f} ➡️ Low Diversity ➡️ Usually binary or heavily skewed (e.g., Gender, Yes/No) ➡️ best action --> Keep. Best for root nodes in decision trees{RESET}')
            elif 1.5 <= entropy_bin < 3.5:
                print(f'{BOLD}{GREEN}Entropy: {entropy_bin:.2f} ➡️ The Sweet Spot ➡️ Healthy clinical variety ➡️ best action --> GOLDEN. This is the best for ML{RESET}')
            elif 3.5 <= entropy_bin < 5:
                print(f'{BOLD}{ORANGE}Entropy: {entropy_bin:.2f} ➡️ High Chaos ➡️ Provides deep detail but can lead to overfitting ➡️ best action --> Group the {n_unique} codes into 5-10 broader categories{RESET}')
            elif entropy_bin > 5:
                print(f'{BOLD}{RED}Entropy: {entropy_bin:.2f} ➡️ White Noise ➡️ Every value is equally likely ➡️ best action ➡️ Drop{RESET}')
            else:
                print('Something wrong with Entropy calculation')
            if 0 <= balance < 0.1:
                print(f'{BOLD}{RED}Balance: {balance:.2f} ➡️ Total Dominance ➡️ One value is essentially a constant. Drop it. It won\'t help the model predict anything{RESET}')
            elif 0.1 <= balance < 0.4:
                print(f'{BOLD}{ORANGE}Balance: {balance:.2f} ➡️ Heavy Imbalance ➡️ Handle with care. You might need to oversample the rare cases{RESET}')
            elif 0.4 <= balance < 0.7:
                print(f'{BOLD}{GREEN}Balance: {balance:.2f} ➡️ Healthy Diversity ➡️ Perfect. This is rich data for modeling{RESET}')
            elif 0.7 <= balance < 0.9:
                print(f'{BOLD}{ORANGE}Balance: {balance:.2f} ➡️ Broad Diversity ➡️ Rich spread across categories. Usually keep; review for fragmentation if many categories are clinically tiny{RESET}')
            elif 0.9 <= balance <= 1:
                print(f'{BOLD}{ORANGE}Balance: {balance:.2f} ➡️ Near-Uniform Spread ➡️ Values are distributed almost equally. Check whether this reflects true diversity or fragmented noise / ID-like categories{RESET}')
            else:
                raise ValueError('❌ Something wrong with Balance calculation')

        def _categorical_graphs(col_name, n_unique):
            if n_unique <= 10:
                plt.figure(figsize=[5,2])
                sns.histplot(x=col_name, data=self.DDA, stat='density')
                plt.xticks(rotation=30, ha="right")
                plt.show()
                plt.close()
            elif n_unique <= 25:
                plt.figure(figsize=[10,2])
                sns.histplot(x=col_name, data=self.DDA, stat='density')
                plt.xticks(rotation=30, ha="right")
                plt.show()
                plt.close()
        # ========== END of HELPER functions ==========

        # === AWESOME CATEGORICAL TABLE ===
        categorical_overview_table_data = []
        # === END of TABLE ===

        for col in categorical_cols:
            # Calculations
            col_name, n, n_unique, missing_pct, first_mode, second_mode, rarest, first_mode_pct, second_mode_pct, max_class_imbalance, median_category, balance, entropy_bin = _categorical_analysis_function(col)
            categorical_overview_table_data.append([col_name, n, n_unique, missing_pct, first_mode, second_mode, rarest, first_mode_pct, second_mode_pct, max_class_imbalance, median_category, balance, entropy_bin])
            # Prints
            _categorical_print(col_name, n, n_unique, missing_pct, first_mode, second_mode, rarest, first_mode_pct, second_mode_pct, max_class_imbalance, median_category, balance, entropy_bin)
            # Graphs
            _categorical_graphs(col_name, n_unique)
        
        categorical_overview_table = pd.DataFrame(
            categorical_overview_table_data,
            columns=[
                'col_name', 'n', 'n_unique', 'missing_pct',
                'first_mode', 'second_mode', 'rarest', 'first_mode_pct', 'second_mode_pct',
                'max_class_imbalance', 'median_category', 'balance', 'entropy_bin'
            ])
        
        numeric_cols = categorical_overview_table.select_dtypes(include="number").columns
        display(categorical_overview_table
                .style
                .format({"n": "{:.0f}", "n_unique": "{:.0f}", "missing_pct": "{:.2f}",
                         "first_mode_pct": "{:.2f}", "second_mode_pct": "{:.2f}", "max_class_imbalance": "{:.2f}",
                         "balance": "{:.2f}", "entropy_bin": "{:.2f}"})
                .background_gradient(
                    subset=numeric_cols,
                    cmap="Reds"))

        self.categorical_DDA = categorical_overview_table
        
        return self
      
    #================= Binary Analysis ====================
    def analyse_binary(self):
        print('\n' + f'{BOLD}==================================================={RESET}')
        print(f'{BOLD}🚦🚦🚦 ================ BINARY =============== 🚦🚦🚦{RESET}')
        print(f'{BOLD}==================================================={RESET}')

        maintenance_cols = ['included_in_cohort', 'row_id', 'cohort_exclusion_reason', 'target']
        binary_cols = [col for col in self.DDA.columns if (len(self.DDA[col].value_counts()) == 2) and col not in maintenance_cols  and col not in self.cols_to_not_analyse]

        if len(binary_cols) == 0:
            print(f'❌ There are no bool/boolean columns in this Dataset')
            return self

        # ============= HELPER functions ==============
        def _binary_analysis_function(col):
            col_to_analyse = self.DDA[col]
            vc = col_to_analyse.value_counts(normalize=True, dropna=True)
            if len(vc) != 2:
                # guard, in case something slipped through
                print(vc)
            len_df = len(col_to_analyse)
            
            #============ THE DATA ============
            cat1 = vc.index[0]                          # most common category
            cat0 = vc.index[1]
            col_name = col
            n = col_to_analyse.count()
            if '_missing' in col_name:
                missing_pct = pd.NA
            else:
                missing_n = col_to_analyse.isna().sum()
                missing_pct = missing_n / len_df
            p1 = vc.iloc[0]                             # already proportion with normalize=True
            p0 = vc.iloc[1]
            if p1 / p0 > 1.1:
                mode = cat1
                mode_pct = p1 * 100
            elif p1 / p0 < 0.9:
                mode = cat0
                mode_pct = p0 * 100
            elif 0.9 <= p1 / p0 <= 1.1:
                mode = 'equal'
                mode_pct = p0 * 100
            balance = 2 * min(p0, p1)
            probs = vc / vc.sum()
            entropy_bin = float(entropy(probs, base=2))
            
            return col_name, cat1, cat0, n, missing_pct, p1, p0, mode, mode_pct, balance, entropy_bin

        def _binary_print(col_name, cat1, cat0, n, missing_pct, p1, p0, mode, mode_pct, balance, entropy_bin):
            print(f'\n{BLUE}{BOLD}===== {col_name} ====={RESET}')
            print(f'{BOLD}n:{RESET} {n}')
            if '_missing' in col_name:
                pass
            else:
                print(f'{BOLD}Missing %:{RESET} {missing_pct:.2f}%')
            print(f'{BOLD}p1 ({cat1}):{RESET} {p1:.2f}')
            print(f'{BOLD}p0 ({cat0}):{RESET} {p0:.2f}')
            print(f'{BOLD}MODE:{RESET} {mode}')
            print(f'{BOLD}MODE %:{RESET} {mode_pct:.2f}%')

            if 0 <= balance < 0.1:
                print(f'{BOLD}{RED}Balance: {balance:.2f} ➡️ Total Dominance - essentially constant. Drop{RESET}')
            elif 0.1 <= balance < 0.3:
                print(f'{BOLD}{YELLOW}Balance: {balance:.2f} ➡️ Heavy Imbalance - rare positives/negatives. Handle with care{RESET}')
            elif 0.3 <= balance < 0.7:
                print(f'{BOLD}{ORANGE}Balance: {balance:.2f} ➡️ Moderate Imbalance - generally fine, but watch rare class{RESET}')
            elif 0.7 <= balance <= 1:
                print(f'{BOLD}{GREEN}Balance: {balance:.2f} ➡️ Healthy Balance - great for modeling{RESET}')
            else:
                raise ValueError('❌ Something wrong with Balance calculation')
            
            if 0.0 <= entropy_bin < 0.05:
                print(f'{BOLD}{RED}Entropy:{entropy_bin:.2f} ➡️ Near-constant - essentially no uncertainty; almost all rows same value. Drop for modeling{RESET}')
            elif 0.05 <= entropy_bin < 0.3:
                print(f'{BOLD}{ORANGE}Entropy:{entropy_bin:.2f} ➡️ Very low entropy - heavily skewed binary (rare positives or negatives). Useful but watch imbalance{RESET}')
            elif 0.3 <= entropy_bin < 0.7:
                print(f'{BOLD}{YELLOW}Entropy:{entropy_bin:.2f} ➡️ Moderate entropy - decent mix, but still some skew. Often fine for modeling{RESET}')
            elif 0.7 <= entropy_bin <= 1.0:
                print(f'{BOLD}{GREEN}Entropy:{entropy_bin:.2f} ➡️ High entropy - close to 50/50. Maximally informative for splits; great for ML{RESET}')
            else:
                print('❌ Something wrong with Entropy calculation')

        def _binary_graph(col):
            plt.figure(figsize=[5,2])
            props = self.DDA[col].value_counts(normalize=True)
            sns.barplot(
                x=props.index.astype(str),
                y=props.values
            )
            plt.ylabel("Proportion")
            plt.xticks(rotation=10, ha="right")
            plt.show()
            plt.close('all')
        # ========== END of HELPER functions ==========

        # === AWESOME BINARY TABLE ===
        binary_overview_table_data = []
        # === END of TABLE ===

        for col in binary_cols:
            # Calculations
            col_name, cat1, cat0, n, missing_pct, p1, p0, mode, mode_pct, balance, entropy_bin = _binary_analysis_function(col)
            binary_overview_table_data.append([col_name, n, missing_pct, p1, p0, mode, mode_pct, balance, entropy_bin])
            # Prints
            _binary_print(col_name, cat1, cat0, n, missing_pct, p1, p0, mode, mode_pct, balance, entropy_bin)
            # Graphs
            _binary_graph(col_name)

        binary_overview_table = pd.DataFrame(
            binary_overview_table_data,
            columns=[
                'column', 'n', 'missing_pct', 'p1', 'p0', 'mode', 'mode_pct', 'balance', 'entropy_bin'
            ]
        )
        
        display(binary_overview_table.style
                .format({"missing_pct": "{:.3f}", "p1": "{:.3f}", "p0": "{:.3f}",
                         "mode_pct": "{:.3f}", "balance": "{:.3f}", "entropy_bin": "{:.3f}"})
                .background_gradient(
                    subset=["missing_pct", "p1", "p0", "mode_pct", "balance", "entropy_bin"],
                    cmap="Reds"))

        self.binary_DDA = binary_overview_table
        return self

    #================= Export Tables ====================
    def export_all_overviews(self, export):
        numerical_path = self.root / 'reports' / 'tables' / f'(numerical_DDA_overview){self.task_name}.pickle'
        categorical_path = self.root / 'reports' / 'tables' / f'(categorical_DDA_overview){self.task_name}.pickle'
        binary_path = self.root / 'reports' / 'tables' / f'(binary_DDA_overview){self.task_name}.pickle'
        if export:
            self.numerical_DDA.to_pickle(numerical_path)
            self.categorical_DDA.to_pickle(categorical_path)
            self.binary_DDA.to_pickle(binary_path)
            print(f'✅ Successful EXPORT of all DDA overviews\n')
            print(f'✅ Folder: {BLUE}{BOLD}ylivertainen/reports/tables/{RESET}')
            print(f'✅ FORMAT: {BLUE}{BOLD}pickle{RESET}\n')
            print(f'✅ Numerical DDA Overview: {BLUE}{BOLD}(numerical_DDA_overview){self.task_name}{RESET}')
            print(f'✅ Categorical DDA Overview: {BLUE}{BOLD}(categorical_DDA_overview){self.task_name}{RESET}')
            print(f'✅ Binary DDA Overview: {BLUE}{BOLD}(binary_DDA_overview){self.task_name}{RESET}')
        else:
            print(f'🔄 Gonna save in: {BLUE}{BOLD}ylivertainen/reports/tables/{RESET}')
            print(f'🔄 FORMAT: {BLUE}{BOLD}pickle{RESET}\n')
            print(f'🔄 Numerical DDA Overview: {BLUE}{BOLD}(numerical_DDA_overview){self.task_name}{RESET}')
            print(f'🔄 Categorical DDA Overview: {BLUE}{BOLD}(categorical_DDA_overview){self.task_name}{RESET}')
            print(f'🔄 Binary DDA Overview: {BLUE}{BOLD}(binary_DDA_overview){self.task_name}{RESET}')
            print(f'===== {BOLD}export={BLUE}True{RESET} to save =====\n')

    def prepare_for_EDA(self):
        print("═" * 70)
        print(f"{BOLD}🧠 DDA STAGE COMPLETE — BRAIN MAP LOCKED IN{RESET}")
        print("─" * 70)

        n_rows, n_cols = self.DDA.shape  # [web:17][web:20]
        total_nans = self.DDA.isna().sum().sum()  # [web:16][web:19]
        nan_pct = (total_nans / (n_rows * n_cols)) * 100 if n_rows and n_cols else 0  # [web:23]

        print(f"{GREEN}✔ DDA FRAME READY FOR EDA ANGIO SUITE{RESET}")
        print(f"   Shape         : {n_rows} rows x {n_cols} cols")
        print(f"   Remaining NaNs: {total_nans} cells ({nan_pct:0.2f}%)")
        print("   Contents      : variable summaries, distributions, missingness map — all scrubbed and labeled.")

        print(f"\n{BOLD}{GREEN}→ Next hop:{RESET} feed `post_DDA_df` straight into the EDA class "
            f"for full association hunting and leakage checks.")

        display(self.DDA.head())
        return self.DDA, self.task_name
