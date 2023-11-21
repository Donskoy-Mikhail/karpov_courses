SELECT sku_type, COUNT(DISTINCT vendor) AS count_vendor
FROM sku_dict_another_one
WHERE not sku_type IS NULL
GROUP BY sku_type
ORDER BY count_vendor DESC
LIMIT 10;
