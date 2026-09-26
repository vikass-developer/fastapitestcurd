-- Week 1 schema for SQL Server. Idempotent: safe to run repeatedly.
-- Batches are separated by GO (split by db.init_db, as SSMS and sqlcmd do).

IF OBJECT_ID(N'dbo.items', N'U') IS NULL
CREATE TABLE dbo.items (
    id          INT            IDENTITY(1, 1) CONSTRAINT pk_items PRIMARY KEY,
    name        NVARCHAR(100)  NOT NULL,
    description NVARCHAR(500)  NULL,
    price       DECIMAL(12, 2) NOT NULL CONSTRAINT ck_items_price CHECK (price > 0),
    created_at  DATETIME2(6)   NOT NULL CONSTRAINT df_items_created_at DEFAULT SYSUTCDATETIME(),
    updated_at  DATETIME2(6)   NULL
);
GO

-- Tags live in their own table (1-to-many) instead of a delimited string column.
IF OBJECT_ID(N'dbo.item_tags', N'U') IS NULL
CREATE TABLE dbo.item_tags (
    item_id  INT          NOT NULL
        CONSTRAINT fk_item_tags_item REFERENCES dbo.items (id) ON DELETE CASCADE,
    position TINYINT      NOT NULL,
    tag      NVARCHAR(50) NOT NULL,
    CONSTRAINT pk_item_tags PRIMARY KEY (item_id, position),
    CONSTRAINT uq_item_tags_item_tag UNIQUE (item_id, tag)
);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'ix_item_tags_tag')
CREATE INDEX ix_item_tags_tag ON dbo.item_tags (tag);
GO

-- Cleaned Titanic dataset (output of `clean-dataset`).
IF OBJECT_ID(N'dbo.passengers', N'U') IS NULL
CREATE TABLE dbo.passengers (
    passenger_id INT           NOT NULL CONSTRAINT pk_passengers PRIMARY KEY,
    survived     BIT           NOT NULL,
    pclass       TINYINT       NOT NULL CONSTRAINT ck_passengers_pclass CHECK (pclass IN (1, 2, 3)),
    name         NVARCHAR(200) NOT NULL,
    sex          NVARCHAR(10)  NOT NULL,
    age          FLOAT         NOT NULL,
    sib_sp       TINYINT       NOT NULL,
    parch        TINYINT       NOT NULL,
    ticket       NVARCHAR(50)  NOT NULL,
    fare         FLOAT         NOT NULL,
    embarked     CHAR(1)       NOT NULL
);
GO

-- Covers the class/sex filters and group-bys without touching the base rows.
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'ix_passengers_pclass_sex')
CREATE INDEX ix_passengers_pclass_sex ON dbo.passengers (pclass, sex) INCLUDE (survived, age);
GO
