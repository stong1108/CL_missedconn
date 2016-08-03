# Craigslist Missed Connections

### Contents
**MissedConn.py**<br>
Class file for a Missed Connections scraper object.

A MissedConn object scrapes Craigslist Missed Connections for a given city. Initialize an object with the url of a city's Craigslist Missed Connections page.

Attributes:
+ `starturl`: url of a Craigslist Missed Connections page used to initialize MissedConn object (e.g. "https://sfbay.craigslist.org/search/mis")
+ `city`: city of the Missed Connections object (e.g. "sfbay")
+ `df`: DataFrame that is populated while scraping. See example at bottom of README.

Methods:
+ `get_df(update=False)`: Populates and returns self.df with Missed Connection post info.

Example:
```
mc = MissedConn('https://sfbay.craigslist.org/search/mis')
df = mc.get_df()
```
*after some time passes, update*
```
df2 = mc.get_df(update=True)
```
***
**manage_db.py**<br>
Contains functions for creating, updating, and retrieving Missed Connections info.

`create_db()`
<br>Creates the database `cl_missedconn`.

`create_table(df)`
<br>Creates the table `missedconn`.

`most_recent_post_dt(city)`
<br>Retrieves the datetime of the most recent post stored in `missedconn` for a given city.

`update_db(df)`
<br>Updates the `missedconn` table with the information in a DataFrame. Only posts that have a unique url are added (no reposts or updated posts).

`db_to_df()`
<br>Creates and outputs a DataFrame representation of `missedconn`.

***
**maps.py**<br>
Contains functions for creating Folium maps to visualize Missed Connections postings. Jupyter notebooks are handy for quick map rendering.

*Note: currently, these maps only display posts where latitude/longitude data were provided*

`make_pinned_map(df, links=False, zoom=12)`
<br>Creates and returns a Folium Map object with popup pins for each post.

`make_heat_map(df, zoom=9)`
<br>Creates and returns a Folium Map object representing a heat map of posts.

***
**posts.py**<br>
*Coming soon*
***
Example `df` attribute of MissedConn object:

![Example Craigslist page](./images/ex_cl_page.png)

|url|title|category|post_dt|latitude|longitude|neighborhood|extra|age|post|record_dt|city|
|:--|:--|:-:|:-:|:-:|:-:|:--|:--|:-:|:--|:-:|:-:|
|https://sfbay.craigslist.org/eby/mis/5673733926.html |WOW!! Delicious woman at Safeway!|m4w|2016-07-08 15:23:00|37.8528|-122.023|Alamo Safeway| |-1|You were coming down the aisle, I moved my car...|2016-07-11 18:04:24|sfbay|
