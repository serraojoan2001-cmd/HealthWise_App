"""
===============================================================================
 FILE : content/qa.py
 ROLE : The lab's reflection questions and the answers from HealthWise_Lab_Joan,
        keyed by task (A1, A2, ... E3, Bonus). Pages import the ones they need.
        Edit the text here and every page updates.
===============================================================================
"""

QA = {
    # ----------------------------------------------------------------- PART A
    "A1": [
        ("A1.1 How many rows and columns are there? What is the unit of one row?",
         "There are **1,203 rows and 17 columns**. Each row is **one customer** (one annual policy record)."),
        ("A1.2 Which columns are numeric and which are categorical?",
         "**Numeric:** age, bmi, weight_kg, children, exercise_freq, lucky_number and annual_charge (target).  \n"
         "**Categorical:** sex, smoker, region, favorite_color, zodiac_sign, preferred_contact, marketing_opt_in "
         "and risk_tier (target).  \n"
         "**Identifier / date:** customer_id and date_of_birth."),
        ("A1.3 Which columns already look suspicious as predictors of medical cost?",
         "**Irrelevant-looking:** lucky_number, favorite_color, zodiac_sign, preferred_contact, marketing_opt_in.  \n"
         "**Redundant:** date_of_birth carries the same information as age, and weight_kg is already inside bmi "
         "(bmi is built from weight and height), so weight_kg can go too."),
    ],
    "A2": [
        ("A2.1 How many duplicate rows did you remove? Why are duplicates dangerous?",
         "**3** duplicate rows (customers HW1005, HW1006, HW1007). Duplicates double-count some customers, which "
         "increases the weight given to them and biases the trained model. They can also land in both the training "
         "and test sets, so test scores look better than they really are."),
        ("A2.2 Which columns had missing values, and how many?",
         "`bmi` and `exercise_freq` had **18 missing values each, 36 in total**."),
        ("A2.3 Justify your imputation choice. Why might median be safer than mean?",
         "The mean can be pulled around by outliers, so the **median** is the safer default when filling gaps. In this "
         "data bmi is nearly symmetric (mean 27.86 vs median 28.0), so the choice barely changes the fill. With only "
         "1.5% of rows affected, imputing costs very little."),
    ],
    "A3": [
        ("A3.1 Is annual_charge symmetric or skewed? What does that imply for a linear model?",
         "It is **right-skewed (skew = 1.15)**: most customers sit at lower dollar values with a long tail of very "
         "expensive ones. A linear model tends to **under-predict the large charges**."),
        ("A3.2 Roughly how much more do smokers cost?",
         "Smokers average $34,511 and non-smokers $14,474, a gap of about **$20,037** a year. That is a very large gap."),
        ("A3.3 Are the three tiers balanced? Why does class balance matter for Stage 1?",
         "No. **Medium is 52%**, Low 26% and High 22%. Balance matters because a lazy classifier could reach ~52% "
         "accuracy just by always saying Medium. What we really care about is not mislabelling High-risk customers as "
         "Medium (under-charging them) or Low-risk customers as Medium (over-charging them)."),
    ],
    "A4": [
        ("A4.1 Which columns did you drop as irrelevant, and why?",
         "**lucky_number, favorite_color, zodiac_sign, preferred_contact and marketing_opt_in.** None of them has a "
         "plausible link to a person's health cost. customer_id was dropped too because it is only a label."),
        ("A4.2 What is lucky_number's correlation with charge, and why is that expected?",
         "**-0.031**, essentially zero. That is expected: a number a person picks at random says nothing about their health."),
        ("A4.3 Some categorical columns show a large spread. What would you check before trusting it?",
         "I would check **how many categories the column has and how many customers sit in each**. zodiac_sign shows "
         "the biggest spread of the junk columns (17.5%), but it has 12 categories with as few as 83 customers each, so "
         "some groups look high or low by chance. I would also run a significance test: the ANOVA p-value for zodiac "
         "is 0.82, so the pattern is noise."),
        ("A4.4 Some remaining columns overlap.",
         "Yes: age vs date_of_birth and bmi vs weight_kg. That is handled in A6."),
    ],
    "A5": [
        ("A5.1 State H0 and H1 for the smoker test.",
         "**H0:** smokers and non-smokers have the same average annual charge.  \n"
         "**H1:** their average annual charge is different."),
        ("A5.2 At alpha = 0.05, reject or fail to reject H0? Explain the p-value in plain business language.",
         "**Reject H0.** The p-value prints as 0.00000 but is really about 2.3 x 10^-84. In business terms: if smoking "
         "made no difference, a gap as large as the observed $20,037 would essentially never happen by luck. The "
         "difference is real."),
        ("A5.3 What does the age-charge correlation add?",
         "r = **0.41** (p is tiny): older customers pay more. Age matters, but on its own it explains only about 17% "
         "of the variation in charge."),
    ],
    "A6": [
        ("A6.1 What are the correlations for bmi/weight_kg and age/date_of_birth?",
         "bmi vs weight_kg = **0.987**. age vs date_of_birth = **1.00**."),
        ("A6.2 Which features have VIF above 5 or 10? Which pair is the worst?",
         "**bmi and weight_kg, both about 37.8.** They are the worst offenders. Every other feature is below 3."),
        ("A6.3 For each redundant pair, which do you keep and why?",
         "**Keep age, drop date_of_birth** (age is given directly; date of birth would need a calculation).  \n"
         "**Keep bmi, drop weight_kg** (bmi already includes weight and height, and is the more interpretable health measure)."),
    ],
    "A7": [
        ("A7.1 What does drop_first=True do, and why does it avoid a new redundancy?",
         "It drops one dummy column per category. That column can always be worked out from the others (the dummies "
         "of one category add up to 1), so keeping it would create a perfect duplicate: the 'dummy variable trap'."),
        ("A7.2 What would underfitting vs overfitting look like for HealthWise's pricing?",
         "**Underfitting (high bias):** a model too simple, e.g. one flat price for all tiers. High-risk customers are "
         "under-priced and low-risk customers over-priced.  \n"
         "**Overfitting (high variance):** a model that learns noise (lucky_number, favorite_color, weight_kg...). "
         "Quotes for new customers become random."),
        ("A7.3 If training error is low but test error is high, which problem is it, and what would you do?",
         "**Overfitting.** Use Ridge or Lasso regularisation, drop noisy columns, or simplify the model."),
    ],
    "A8": [
        ("A8.1 As alpha increases, which features survive and which are zeroed?",
         "At alpha = 800 only **age, bmi, weight_kg and smoker_yes** survive. The junk columns are zeroed. (At "
         "alpha = 400 two junk dummies, favorite_color_green and zodiac_sign_Pisces, are still barely present.)"),
        ("A8.2 Does R-squared fall much when the junk is removed? What does that prove?",
         "**No.** R-squared is 0.684 with all 29 features and 0.685 with 4. The junk columns contributed nothing."),
        ("A8.3 Did Lasso agree with your manual audit?",
         "**Mostly.** Lasso independently threw out the same junk columns. Two differences: it kept both bmi and "
         "weight_kg (Lasso does not reliably remove correlated pairs, so the VIF check was still needed), and at "
         "alpha = 800 it also dropped sex, children, region and exercise_freq, which I kept because they are plausible real factors."),
    ],
    # ----------------------------------------------------------------- PART B
    "B1": [
        ("B1.1 Which classifier did you choose and why?",
         "**Logistic Regression** for the evaluation in Part B: its coefficients show each feature's impact clearly and "
         "it gives class probabilities, which are useful in a customer-facing app. (The app itself deploys a Decision "
         "Tree, as the lab's task E1 prescribes. The page compares the two.)"),
        ("B1.2 Why do we use stratify=y_cls?",
         "The tiers are imbalanced (Medium is the majority). Stratifying keeps the same tier proportions in the train "
         "and test sets, so the test set is a fair sample and High-risk customers are not under-represented."),
    ],
    "B2": [
        ("B2.1 What is your test accuracy?", "**0.821** for Logistic Regression."),
        ("B2.2 Which tier is most confused with which?",
         "**Actual High predicted as Medium** (16 customers) is the biggest error. Then **actual Low predicted as "
         "Medium** (12) and **actual Medium predicted as Low** (11). Low and High are almost never confused with each other."),
        ("B2.3 Business risk: which is worse, High labelled Low, or Low labelled High?",
         "**High labelled Low is worse.** HealthWise would collect a low premium from someone who will file expensive "
         "claims, losing money year after year. **Low labelled High** overcharges a safe customer, who may leave for a "
         "competitor. That costs revenue, but far less."),
    ],
    "B3": [
        ("B3.1 Is there a large gap between train and test accuracy?",
         "No: train 0.785 and test 0.821, a difference of 0.035, and test is even slightly higher."),
        ("B3.2 Overfitting, underfitting, or well-balanced?",
         "**Well-balanced.** Train and test are close. Because accuracy is only around 80%, the model could still be "
         "improved (the Decision Tree on the same page shows how much)."),
    ],
    # ----------------------------------------------------------------- PART C
    "C1": [
        ("C1.1 Record the global MAE, RMSE and R-squared.", "**MAE $5,698, RMSE $7,518, R-squared 0.691.**"),
        ("C1.2 How good does one model for everyone look?",
         "It explains about 69% of the variation, but a typical quote is off by about $5,700. High-risk customers get "
         "undercharged and low-risk customers overcharged, so one formula for everyone is not good enough."),
    ],
    "C2": [
        ("C2.1 How much did R-squared improve from one feature to all features?",
         "**High tier +0.625** (0.129 to 0.754, using bmi). **Low tier +0.394** (0.407 to 0.801, using age)."),
        ("C2.2 Compare the jump for Low and High. Which is bigger, and why?",
         "The jump is **bigger for High-risk customers**. Their charge depends on several features together (bmi, "
         "smoker, age, exercise), while in the Low tier a single feature (age) already explains a lot."),
        ("C2.3 Does every added feature deserve credit? How would you check?",
         "No. More features always help the training fit, so we might credit noise. Lasso (C4) checks this by "
         "shrinking useless coefficients to zero."),
    ],
    "C3": [
        ("C3.1 Fill in the global vs two-stage table.",
         "| Model | MAE | R-squared |\n|---|---|---|\n| Global (C1) | $5,698 | 0.691 |\n"
         "| Two-stage per tier (C3, perfect tier) | $908 | 0.992 |"),
        ("C3.2 Board question 1: does two-stage beat the global model? Quote the MAE improvement.",
         "**Yes.** MAE falls by **$4,790 (84%)** when each customer is priced by the correct tier's model. "
         "*(Caveat added in this dashboard: that assumes Stage 1 always picks the right tier. With the deployed Decision "
         "Tree choosing the tier, the honest improvement is about 72%. See the 'Honest test' tab.)*"),
        ("C3.3 Which tier fits best and which worst?",
         "**Low fits best** (MAE $378, R-squared 0.801). **High fits worst** (MAE $1,919, R-squared 0.754). Low-risk "
         "customers are the most predictable; High-risk customers the least."),
    ],
    "C4": [
        ("C4.1 Which alpha gave the best test R-squared? What happens when alpha is too large?",
         "Ridge is flat up to alpha = 1 and Lasso stays at 0.754 all the way to alpha = 50, so alpha = 1 (Ridge) and "
         "10 (Lasso) are good choices. When alpha gets too large, the penalty shrinks coefficients too much and the "
         "model **underfits** (Ridge falls to 0.457 at alpha = 200)."),
        ("C4.2 Which real features drive cost in this tier? Does region matter?",
         "In the High tier, in order: **bmi, smoker_yes, age, children, sex_male**. Region hardly matters: its "
         "coefficients are tiny next to the top drivers. exercise_freq was zeroed because it overlaps with bmi."),
        ("C4.3 Why is regularisation useful even after removing the obvious junk?",
         "It gently controls weak or overlapping coefficients. For example bmi and exercise_freq are correlated, which "
         "can make coefficients unstable, and each tier has few customers. (Here it did not raise R-squared, since the "
         "data is clean, but it confirms the ranking of drivers and makes the model safer on new data.)"),
    ],
    "C5": [
        ("C5.1 Complete the model ladder (High tier).",
         "| Model | R-squared | What it tells you |\n|---|---|---|\n"
         "| Simple linear (bmi) | 0.129 | One feature explains little |\n"
         "| Multiple linear | 0.754 | Several real features together explain most of the variance |\n"
         "| Ridge (tuned) | 0.754 | No accuracy gain |\n| Lasso (tuned) | 0.754 | Same accuracy, but drops exercise_freq |"),
        ("C5.2 Where was the biggest gain?",
         "**Adding the real features** (+0.62 R-squared) and **splitting customers into tiers** (much lower dollar "
         "error). Cleaning out junk columns and tuning alpha changed accuracy very little (A8 showed R-squared "
         "0.684 vs 0.685). The problem is driven by a small set of genuine features acting together."),
    ],
    # ----------------------------------------------------------------- PART D
    "D": [
        ("D.1 Does the two-stage engine beat the flat model?",
         "Yes. Average pricing error fell from $5,698 to $908 and R-squared rose from 0.69 to 0.99 (lab C3, perfect "
         "tier). With the deployed classifier choosing the tier, the honest numbers are $6,167 to $1,706 and 0.72 to 0.89."),
        ("D.2 Which attributes drive cost and which are worthless?",
         "**Top three drivers: smoker, BMI and age.** Dropped as irrelevant: lucky_number, favorite_color, "
         "zodiac_sign, preferred_contact, marketing_opt_in. Dropped as redundant: date_of_birth and weight_kg. "
         "customer_id was dropped as an identifier."),
        ("D.3 Recommendation: deploy or not? One risk or limitation.",
         "**Deploy**, with the tier-classification step monitored. The Logistic model in Part B was right 82% of the "
         "time; the deployed Decision Tree is right about 94% on held-out customers. Most of its mistakes put a "
         "High-risk customer into Medium, which under-prices them."),
    ],
    # ----------------------------------------------------------------- PART E
    "E1": [
        ("E1.1 Why must the same encoder be used at training and prediction time?",
         "The model only understands the exact column layout it was trained on. A single customer row contains only "
         "one region and one sex, so a plain get_dummies would create the wrong columns. Reusing the fitted encoder "
         "guarantees identical columns in identical order."),
        ("E1.2 Why save three regressors plus one classifier, rather than one model?",
         "Each tier has its own cost structure. For example, being a smoker adds about $2,756 in the Low tier, $4,248 "
         "in Medium and $8,111 in High. The classifier picks the tier and that tier's regressor prices the customer."),
    ],
    "E2": [
        ("E2.1 Show a prediction.",
         "Open the **Price Lab** page: default customer (age 40, BMI 27, 1 child, 3 workouts, male, non-smoker, north) "
         "is quoted **Low risk, $8,101**."),
        ("E2.2 Start from a Low-risk customer and switch smoker to yes. What happens, and why?",
         "The tier jumps **Low to Medium** and the premium rises from about **$8,100 to $18,755** for the default "
         "customer. Stage 1 sees the smoker flag and moves the customer up a tier; Stage 2 then prices them with the "
         "Medium regression, which has a higher base rate and a bigger smoker effect. Smoking raises the price twice."),
        ("E2.3 Which single change moves the premium the most: smoker, age, or bmi?",
         "**Smoker.** It moved the premium most in every profile I tried. That matches the C4 coefficients: smoker is "
         "one of the top two drivers. Note that a large part of any jump comes from the customer crossing a tier "
         "boundary, not from the coefficient itself."),
    ],
    "E3": [
        ("E3 Score the unseen customers.",
         "Use the **Batch Scorer** page: upload the instructor's `test_customers.csv`, then apply the named change "
         "(e.g. smoker to yes, or +2 children) and copy the table into the notebook."),
    ],
    "Bonus": [
        ("Bonus 1: A truly High-risk customer is misclassified as Low. What happens, and what safeguard would you add?",
         "They are priced by the Low-tier regression, so the quote is far below their true cost and HealthWise loses "
         "money on every claim. Safeguards: send low-confidence classifications to manual review, use asymmetric "
         "thresholds so borderline cases lean towards the higher tier, and monitor High-to-Low errors separately."),
        ("Bonus 2: When would you use PCA instead?",
         "PCA is useful when there are **many correlated features** (dozens or hundreds) and you mainly want "
         "prediction accuracy or a compact representation. The trade-off is **interpretability**: components are "
         "blends of the original columns, so you can no longer say what smoking or BMI is worth in dollars. Here we "
         "had few useful features and needed explainable prices, so VIF and Lasso were the better fit."),
    ],
}
