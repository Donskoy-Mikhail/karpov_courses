SELECT
    age,
    income,
    dependents,
    has_property,
    has_car,
    credit_score,
    job_tenure,
    has_education,
    loan_amount,
    datediff('day', loan_start, loan_deadline) AS loan_period,
    CASE
        WHEN datediff('day', loan_deadline, loan_payed) < 0 THEN 0
        ELSE datediff('day', loan_deadline, loan_payed)
    END AS  delay_days
FROM default.loan_delay_days
ORDER BY id;