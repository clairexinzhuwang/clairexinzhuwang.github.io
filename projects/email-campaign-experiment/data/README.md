# Data provenance

This project uses Kevin Hillstrom's public **MineThatData E-Mail Analytics and
Data Mining Challenge** dataset, released on March 20, 2008.

- Source description: <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>
- Original CSV: <http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv>
- Expected rows: 64,000 customers, excluding the header
- Expected SHA-256: `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`

The source describes a randomized three-arm experiment: approximately one
third of customers received a men's merchandise email, one third received a
women's merchandise email, and one third received no email. Visit, conversion,
and spend were measured over the following two weeks.

The original page does not state a formal data license. For that reason this
repository does **not** redistribute the customer-level CSV. The download
script retrieves the public source for local analysis and verifies the exact
file hash. Anyone redistributing the raw data should independently confirm
that their use is permitted.

Run from the repository root:

```bash
python3 -m src.download_data
```

The file will be stored at `data/raw/hillstrom.csv`. The entire `data/raw/`
directory is excluded from version control.

## Source fields

The released columns are pre-treatment recency and purchase-history features;
the randomized `segment`; and the post-treatment outcomes `visit`,
`conversion`, and `spend`. The original file spells the suburban zip category
as `Surburban`. The analysis records that source issue and maps it to
`Suburban` in memory without changing the downloaded file.

There is no customer identifier. Exact duplicate rows can therefore be
counted, but cannot be interpreted as duplicate customers and are never
dropped.

