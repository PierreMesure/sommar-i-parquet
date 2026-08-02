# Advice

## Exploring Wikidata

You have access to MCP tools to navigate Wikidata but unfortunately, the search_items tool is not that good. Use execute_sparql() and get_statement_values() instead, you will have more possibilities to filter and less stale cache.

## Submitting QuickStatements

When submitting QuickStatements batches, use the API and the `QS_TOKEN` in [.env](./.env)

Use the token to submit a batch to QuickStatements from your own bot (please use POST):

> ./api.php
> ?action=import
> &submit=1
> &username=PierreMesure
> &token=%242y%2412%24W.ULRE1xa7Bd1fu5oRKqq.EkU5DXJzjQeBW8KknVv49l4rqgthLb.
> &format=FORMAT ["v1" or "csv"]
> &data=COMMANDS [commands in the above format]
> &compress=0 [optional; deactivates compression of CREATE and following LAST commands]
> &batchname=BATCH_NAME [optional]
> &site=SITE_KEY [optional; default:"wikidata"]

The batch will start automatically, as if you had created it in the interface and then clicked "Run in background".
A JSON object will be returned: {"status":"OK","batch_id":ID_OF_THE_NEW_BATCH} (or an error message in "status").
