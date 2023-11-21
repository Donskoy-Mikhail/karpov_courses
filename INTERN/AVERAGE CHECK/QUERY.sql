SELECT
    toStartOfMonth(buy_date::DATE) AS month,
    AVG(check_amount) AS avg_check,
    quantileExact(0.5)(check_amount::double) AS median_check
FROM
    default.view_checks
GROUP BY
    month
ORDER BY
    month;