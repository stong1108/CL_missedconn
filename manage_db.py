from pandas.io.sql import read_sql
from sqlalchemy import create_engine
from langdetect import detect
import pickle

def create_db():
    '''
    Creates the database 'cl_missedconn'.
    '''
    engine = create_engine('postgresql://stong@localhost:5432/postgres')
    conn = engine.connect()
    conn.connection.connection.set_isolation_level(0)
    conn.execute('CREATE DATABASE cl_missedconn')
    conn.connection.connection.set_isolation_level(1)
    conn.close()

def create_table(df):
    '''
    Creates the table 'missedconn'.

    INPUT:
        - df (DataFrame): DataFrame output from the MissedConn method 'get_df()'
    '''
    engine = create_engine('postgresql://stong@localhost:5432/cl_missedconn')
    conn = engine.connect()
    conn.execute(
        '''
        CREATE TABLE missedconn(
            url TEXT PRIMARY KEY NOT NULL,
            title TEXT NOT NULL,
            category TEXT,
            post_dt TIMESTAMP NOT NULL,
            latitude NUMERIC,
            longitude NUMERIC,
            neighborhood TEXT,
            extra TEXT,
            age INT NOT NULL,
            post TEXT,
            record_dt TIMESTAMP NOT NULL,
            city TEXT,
            raw_page TEXT,
            has_pic INT
        );
        '''
    )

    df.set_index('url', inplace=True)
    df.to_sql('missedconn', engine, if_exists='append')
    df.reset_index(inplace=True)

    conn.close()

def most_recent_post_dt(city):
    '''
    Retrieves the datetime of the most recent post stored in 'missedconn'
    for a given city.

    INPUT:
        - city (str): name of city in Craigslist url (e.g. "sfbay"). Can pass
        a MissedConn's attribute 'city' (e.g. mc.city).

    OUTPUT:
        - recent_dt (datetime): datetime of most recent 'post_dt' for city in
        'missedconn'
    '''
    engine = create_engine('postgresql://stong@localhost:5432/cl_missedconn')
    conn = engine.connect()
    query_result = conn.execute('SELECT MAX(post_dt) FROM missedconn WHERE city = (%s)', (city,))
    recent_dt = query_result.fetchone()[0]
    conn.close()
    return recent_dt

def update_db(df):
    '''
    Updates the missedconn table with the information in a DataFrame. Only posts
    that have a unique url are added (no reposts or updated posts).

    INPUT:
        - df (DataFrame): DataFrame output from the MissedConn method 'get_df(update=True)'
    '''
    engine = create_engine('postgresql://stong@localhost:5432/cl_missedconn')
    conn = engine.connect()

    df.set_index('url', inplace=True)
    df.to_sql('tmp', engine)
    df.reset_index(inplace=True)

    # Delete multiple posts that have same content (determined by title)
    # These posts may have different urls and might not be caught in the upload
    conn.execute(
        '''
        DELETE FROM tmp
        WHERE EXISTS
        (SELECT 1 FROM tmp a
        WHERE (a.url = tmp.url
        OR a.title = tmp.title)
        AND a.ctid > tmp.ctid)
        '''
    )

    # Only add posts with urls or titles that don't already exist in table
    conn.execute(
        '''
        INSERT INTO missedconn
        SELECT * FROM tmp
        WHERE NOT EXISTS
        (SELECT 1 FROM missedconn
        WHERE url = tmp.url)
        AND NOT EXISTS
        (SELECT 1 FROM missedconn
        WHERE title = tmp.title)
        '''
    )
    conn.execute('DROP TABLE tmp')
    conn.close()

def update_db2(df):
    '''
    Updates the missedconn table with the information in a DataFrame. Only posts
    that have a unique url are added (no reposts or updated posts).

    INPUT:
        - df (DataFrame): DataFrame output from the MissedConn method 'get_df(update=True)'
    '''
    engine = create_engine('postgresql://stong@localhost:5432/cl_missedconn')
    conn = engine.connect()

    df.set_index('url', inplace=True)
    df.to_sql('tmp', engine)
    df.reset_index(inplace=True)

    # Only add posts with urls or titles that don't already exist in table
    conn.execute(
        '''
        INSERT INTO missedconn
        SELECT * FROM tmp
        WHERE NOT EXISTS
        (SELECT 1 FROM missedconn
        WHERE url = tmp.url)
        AND NOT EXISTS
        (SELECT 1 FROM missedconn
        WHERE title = tmp.title)
        '''
    )
    conn.execute('DROP TABLE tmp')
    conn.close()

def db_to_df():
    '''
    Creates and outputs a DataFrame representation of 'missedconn'.

    OUTPUT:
        - df (DataFrame): DataFrame representation of 'missedconn' with the
        same structure as the MissedConn attribute 'df'.
    '''
    engine = create_engine('postgresql://stong@localhost:5432/cl_missedconn')
    conn = engine.connect()
    df = read_sql('SELECT * FROM missedconn', conn, index_col='url')
    df.reset_index(inplace=True)
    conn.close()
    return df

def make_english_pickle(picklename):
    '''
    Creates a pickle object containing English posts only
    '''
    with open('bestofmc.pickle', 'rb') as f:
        df_best = pickle.load(f)

    df_all = db_to_df()

    df = df_all.append(df_best)
    df.reset_index(inplace=True)
    df['text'] = map(lambda x,y: ' '.join([x, y]), df['title'], df['post'])
    eng_inds = [i for i in xrange(len(df)) if detect(df.loc[i, 'text']) == 'en']

    with open(picklename, 'wb') as f:
        pickle.dump(df.loc[eng_inds], f)
