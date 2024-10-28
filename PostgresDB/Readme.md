extraction.py => a python file to extract fall or not-fall data into csv
populate-database => a python file that assumes you have all of the text files provided in [canada.ca](https://www.canada.ca/en/health-canada/services/drugs-health-products/medeffect-canada/adverse-reaction-database/canada-vigilance-online-database-data-extract.html) in the same directory and populates a postgres database. (change the configuration for your local postgres database if needed). Also make sure that you have unzipped the zip file provided in the previous link

for extraction, the current state gets info for not-fall. I used the same script for fall with slight changes.
change the first two queries to use NOT EXIST instead of EXISTS, (leave the last query)
then set the variable "target" = 1

init_db.sql => should be run in the beginning to set up postgres schema
