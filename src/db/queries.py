"""The five analytics queries over dbo.passengers.

Each one shows a different SQL building block. Parameters use `?` placeholders, so values are
never formatted into the SQL text.
"""

# 1. GROUP BY on two columns with aggregates (COUNT, SUM, AVG).
SURVIVAL_BY_CLASS_AND_SEX = """
SELECT pclass,
       sex,
       COUNT(*)                                       AS passengers,
       SUM(CAST(survived AS INT))                     AS survivors,
       ROUND(AVG(CAST(survived AS FLOAT)) * 100, 1)   AS survival_rate_pct
FROM dbo.passengers
GROUP BY pclass, sex
ORDER BY pclass, sex;
"""

# 2. CTE plus CASE to bucket a continuous column, with a sort key to keep the buckets in order.
SURVIVAL_BY_AGE_GROUP = """
WITH bucketed AS (
    SELECT survived,
           CASE WHEN age < 13 THEN 1 WHEN age < 20 THEN 2 WHEN age < 40 THEN 3
                WHEN age < 60 THEN 4 ELSE 5 END AS sort_key,
           CASE WHEN age < 13 THEN 'child (0-12)'
                WHEN age < 20 THEN 'teen (13-19)'
                WHEN age < 40 THEN 'adult (20-39)'
                WHEN age < 60 THEN 'middle-aged (40-59)'
                ELSE 'senior (60+)' END AS age_group
    FROM dbo.passengers
)
SELECT age_group,
       COUNT(*)                                       AS passengers,
       SUM(CAST(survived AS INT))                     AS survivors,
       ROUND(AVG(CAST(survived AS FLOAT)) * 100, 1)   AS survival_rate_pct
FROM bucketed
GROUP BY sort_key, age_group
ORDER BY sort_key;
"""

# 3. Window function: top-N rows per group with ROW_NUMBER() OVER (PARTITION BY ...).
#    Param: top_n.
OLDEST_PER_CLASS = """
WITH ranked AS (
    SELECT passenger_id, name, pclass, age, survived,
           ROW_NUMBER() OVER (PARTITION BY pclass ORDER BY age DESC, passenger_id) AS age_rank
    FROM dbo.passengers
)
SELECT pclass, age_rank, passenger_id, name, age, survived
FROM ranked
WHERE age_rank <= ?
ORDER BY pclass, age_rank;
"""

# 4. GROUP BY a derived expression, with HAVING to drop tiny groups. Param: min_passengers.
SURVIVAL_BY_FAMILY_SIZE = """
SELECT sib_sp + parch + 1                             AS family_size,
       COUNT(*)                                       AS passengers,
       SUM(CAST(survived AS INT))                     AS survivors,
       ROUND(AVG(CAST(survived AS FLOAT)) * 100, 1)   AS survival_rate_pct
FROM dbo.passengers
GROUP BY sib_sp + parch + 1
HAVING COUNT(*) >= ?
ORDER BY family_size;
"""

# 5. JOIN to an inline lookup table (VALUES), with share of total via SUM() OVER ().
EMBARKED_SUMMARY = """
SELECT p.embarked,
       ports.port_name,
       COUNT(*)                                             AS passengers,
       ROUND(AVG(p.fare), 2)                                AS avg_fare,
       ROUND(AVG(CAST(p.survived AS FLOAT)) * 100, 1)       AS survival_rate_pct,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)   AS share_pct
FROM dbo.passengers AS p
JOIN (VALUES ('C', 'Cherbourg'), ('Q', 'Queenstown'), ('S', 'Southampton'))
     AS ports (code, port_name) ON ports.code = p.embarked
GROUP BY p.embarked, ports.port_name
ORDER BY passengers DESC;
"""
