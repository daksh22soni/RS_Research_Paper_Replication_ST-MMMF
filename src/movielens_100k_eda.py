"""
MovieLens 100k Dataset Analysis
Fixed for: u.item, u.data, u.user only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("MOVIELENS 100K DATASET ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: UNDERSTAND FILE PURPOSE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: FILE PURPOSE UNDERSTANDING")
print("=" * 80)

file_purposes = {
    'u.item': 'Movie information (ID, title, release date, genres as binary flags)',
    'u.data': 'User ratings (userId, movieId, rating 1-5, Unix timestamp)',
    'u.user': 'User demographics (userId, age, gender, occupation, zip)',
}

for file, purpose in file_purposes.items():
    print(f"📁 {file:15} → {purpose}")

# ============================================================================
# STEP 2: BASIC STRUCTURE CHECK
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: BASIC STRUCTURE CHECK")
print("=" * 80)

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

print("\n📂 Loading datasets...")

movies = pd.read_csv(
    r"C:/Users/gahan/Documents/Daksh/sem 6/Recommendation System/project/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    names=["movieId", "title", "release_date", "video_release_date", "IMDb_URL"] + GENRE_COLS,
)

ratings = pd.read_csv(
    r"C:/Users/gahan/Documents/Daksh/sem 6/Recommendation System/project/ml-100k/u.data",
    sep="\t",
    header=None,
    names=["userId", "movieId", "rating", "timestamp"],
)

users = pd.read_csv(
    r"C:/Users/gahan/Documents/Daksh/sem 6/Recommendation System/project/ml-100k/u.user",
    sep="|",
    header=None,
    names=["userId", "age", "gender", "occupation", "zip"],
)

print("✅ All files loaded successfully!\n")

datasets = {
    'u.item  (movies)': movies,
    'u.data  (ratings)': ratings,
    'u.user  (users)': users,
}

for name, df in datasets.items():
    print(f"📊 {name}")
    print(f"   Rows    : {len(df):,}")
    print(f"   Columns : {list(df.columns)}")
    print(f"   Shape   : {df.shape}")
    print(f"   Memory  : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

# ============================================================================
# STEP 3: MISSING VALUES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: MISSING VALUES ANALYSIS")
print("=" * 80)

for name, df in datasets.items():
    print(f"\n📋 {name}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})

    if missing.sum() > 0:
        print(missing_df[missing_df['Missing Count'] > 0].to_string())
        print(f"   ⚠️  Total missing values: {missing.sum():,}")
    else:
        print("   ✅ No missing values!")

# ============================================================================
# STEP 4: RATING DISTRIBUTION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: RATING DISTRIBUTION")
print("=" * 80)

rating_dist = ratings['rating'].value_counts().sort_index()
print("\n📊 Rating Distribution:")
print(rating_dist.to_string())

print(f"\n📈 Statistics:")
print(f"   Mean rating    : {ratings['rating'].mean():.2f}")
print(f"   Median rating  : {ratings['rating'].median():.2f}")
print(f"   Std deviation  : {ratings['rating'].std():.2f}")
print(f"   Min rating     : {ratings['rating'].min():.1f}")
print(f"   Max rating     : {ratings['rating'].max():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(ratings['rating'], bins=20, edgecolor='black', color='skyblue')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Rating Distribution (Histogram)')
axes[0].grid(axis='y', alpha=0.3)

axes[1].boxplot(ratings['rating'], vert=True)
axes[1].set_ylabel('Rating')
axes[1].set_title('Rating Distribution (Box Plot)')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('step4_rating_distribution.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: step4_rating_distribution.png")
plt.close()

# ============================================================================
# STEP 5: USER ACTIVITY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: USER ACTIVITY ANALYSIS")
print("=" * 80)

ratings_per_user = ratings.groupby('userId').size()

print(f"\n👥 User Activity Statistics:")
print(f"   Total unique users        : {ratings['userId'].nunique():,}")
print(f"   Mean ratings per user     : {ratings_per_user.mean():.2f}")
print(f"   Median ratings per user   : {ratings_per_user.median():.0f}")
print(f"   Min ratings per user      : {ratings_per_user.min()}")
print(f"   Max ratings per user      : {ratings_per_user.max()}")
print(f"   Std deviation             : {ratings_per_user.std():.2f}")

print(f"\n🔝 Top 10 Most Active Users:")
top_users = ratings_per_user.sort_values(ascending=False).head(10)
for idx, (user_id, count) in enumerate(top_users.items(), 1):
    print(f"   {idx:2}. User {user_id:5} → {count:4} ratings")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(ratings_per_user, bins=50, edgecolor='black', color='coral')
axes[0].set_xlabel('Number of Ratings')
axes[0].set_ylabel('Number of Users')
axes[0].set_title('User Activity Distribution')
axes[0].set_yscale('log')
axes[0].grid(axis='y', alpha=0.3)

sorted_ratings = np.sort(ratings_per_user)
cumulative = np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings) * 100
axes[1].plot(sorted_ratings, cumulative, color='coral', linewidth=2)
axes[1].set_xlabel('Number of Ratings')
axes[1].set_ylabel('Cumulative % of Users')
axes[1].set_title('Cumulative User Activity')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('step5_user_activity.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: step5_user_activity.png")
plt.close()

# ============================================================================
# STEP 6: MOVIE POPULARITY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MOVIE POPULARITY ANALYSIS")
print("=" * 80)

ratings_per_movie = ratings.groupby('movieId').size()

print(f"\n🎬 Movie Popularity Statistics:")
print(f"   Total unique movies rated   : {ratings['movieId'].nunique():,}")
print(f"   Total movies in catalog     : {len(movies):,}")
print(f"   Movies never rated          : {len(movies) - ratings['movieId'].nunique():,}")
print(f"   Mean ratings per movie      : {ratings_per_movie.mean():.2f}")
print(f"   Median ratings per movie    : {ratings_per_movie.median():.0f}")
print(f"   Min ratings per movie       : {ratings_per_movie.min()}")
print(f"   Max ratings per movie       : {ratings_per_movie.max()}")

print(f"\n🌟 Top 10 Most Rated Movies:")
top_movies = ratings_per_movie.sort_values(ascending=False).head(10)
for idx, (movie_id, count) in enumerate(top_movies.items(), 1):
    movie_title = movies.loc[movies['movieId'] == movie_id, 'title'].values[0]
    print(f"   {idx:2}. {movie_title:50} → {count:4} ratings")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(ratings_per_movie, bins=50, edgecolor='black', color='lightgreen')
axes[0].set_xlabel('Number of Ratings')
axes[0].set_ylabel('Number of Movies')
axes[0].set_title('Movie Popularity Distribution')
axes[0].set_yscale('log')
axes[0].grid(axis='y', alpha=0.3)

top_20_movies = ratings_per_movie.sort_values(ascending=False).head(20)
movie_titles = []
for mid in top_20_movies.index:
    t = movies.loc[movies['movieId'] == mid, 'title'].values[0]
    movie_titles.append(t[:30] + '...' if len(t) > 30 else t)

axes[1].barh(range(len(top_20_movies)), top_20_movies.values, color='lightgreen')
axes[1].set_yticks(range(len(top_20_movies)))
axes[1].set_yticklabels(movie_titles, fontsize=8)
axes[1].set_xlabel('Number of Ratings')
axes[1].set_title('Top 20 Most Rated Movies')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('step6_movie_popularity.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: step6_movie_popularity.png")
plt.close()

# ============================================================================
# STEP 7: GENRE DISTRIBUTION  (binary columns in u.item)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: GENRE DISTRIBUTION ANALYSIS")
print("=" * 80)

# u.item stores genres as 19 binary columns — sum each column
genre_counts = movies[GENRE_COLS].sum().sort_values(ascending=False)

print(f"\n🎭 Genre Statistics:")
print(f"   Total unique genres        : {len(genre_counts)}")
print(f"   Total genre assignments    : {int(genre_counts.sum())}")

print(f"\n📊 Genre Distribution:")
for genre, count in genre_counts.items():
    pct = count / len(movies) * 100
    print(f"   {genre:20} → {int(count):5} movies ({pct:5.1f}%)")

# Multi-genre movies (more than one genre flag set)
genre_per_movie = movies[GENRE_COLS].sum(axis=1)
multi_genre = (genre_per_movie > 1).sum()
print(f"\n🎬 Genre Combinations:")
print(f"   Movies with multiple genres : {int(multi_genre):,} ({multi_genre/len(movies)*100:.1f}%)")
print(f"   Movies with single genre    : {int((genre_per_movie == 1).sum()):,}")
print(f"   Movies with no genre listed : {int((genre_per_movie == 0).sum()):,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(genre_counts.index, genre_counts.values, color='mediumpurple')
axes[0].set_xlabel('Number of Movies')
axes[0].set_title('Genre Distribution')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

top_10_genres = genre_counts.head(10)
axes[1].pie(top_10_genres.values, labels=top_10_genres.index,
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Top 10 Genres Distribution')

plt.tight_layout()
plt.savefig('step7_genre_distribution.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: step7_genre_distribution.png")
plt.close()

# ============================================================================
# STEP 8: USER DEMOGRAPHICS  (replaces tag analysis — not in ml-100k)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: USER DEMOGRAPHICS ANALYSIS")
print("=" * 80)

print(f"\n👤 Total users: {len(users):,}")

print(f"\n🔢 Age Statistics:")
print(f"   Mean age   : {users['age'].mean():.1f}")
print(f"   Median age : {users['age'].median():.0f}")
print(f"   Min age    : {users['age'].min()}")
print(f"   Max age    : {users['age'].max()}")

gender_dist = users['gender'].value_counts()
print(f"\n⚤ Gender Distribution:")
for gender, count in gender_dist.items():
    print(f"   {gender} : {count:,} ({count/len(users)*100:.1f}%)")

occ_dist = users['occupation'].value_counts()
print(f"\n💼 Top 10 Occupations:")
for idx, (occ, count) in enumerate(occ_dist.head(10).items(), 1):
    print(f"   {idx:2}. {occ:20} → {count:4} users")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age histogram
axes[0, 0].hist(users['age'], bins=20, edgecolor='black', color='steelblue')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Number of Users')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].grid(axis='y', alpha=0.3)

# Gender pie
axes[0, 1].pie(gender_dist.values, labels=gender_dist.index,
               autopct='%1.1f%%', startangle=90, colors=['cornflowerblue', 'lightcoral'])
axes[0, 1].set_title('Gender Distribution')

# Occupation bar
top_occs = occ_dist.head(15)
axes[1, 0].barh(top_occs.index, top_occs.values, color='mediumseagreen')
axes[1, 0].set_xlabel('Number of Users')
axes[1, 0].set_title('Top 15 Occupations')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# Age by gender box plot
male_ages = users.loc[users['gender'] == 'M', 'age']
female_ages = users.loc[users['gender'] == 'F', 'age']
axes[1, 1].boxplot([male_ages, female_ages], labels=['Male', 'Female'])
axes[1, 1].set_ylabel('Age')
axes[1, 1].set_title('Age Distribution by Gender')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('step8_user_demographics.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: step8_user_demographics.png")
plt.close()

# ============================================================================
# STEP 9: RATING TRENDS OVER TIME  (replaces links completeness)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: RATING TRENDS OVER TIME")
print("=" * 80)

ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['year_month'] = ratings['datetime'].dt.to_period('M')

monthly_counts = ratings.groupby('year_month').size()
monthly_avg = ratings.groupby('year_month')['rating'].mean()

print(f"\n📅 Time Range:")
print(f"   Earliest rating : {ratings['datetime'].min().date()}")
print(f"   Latest rating   : {ratings['datetime'].max().date()}")
print(f"   Total months    : {len(monthly_counts)}")
print(f"\n📊 Monthly Activity:")
print(f"   Mean ratings/month  : {monthly_counts.mean():.0f}")
print(f"   Peak month          : {monthly_counts.idxmax()} ({monthly_counts.max():,} ratings)")
print(f"   Slowest month       : {monthly_counts.idxmin()} ({monthly_counts.min():,} ratings)")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

x = range(len(monthly_counts))
axes[0].bar(x, monthly_counts.values, color='steelblue', alpha=0.8)
axes[0].set_xticks(x[::max(1, len(x)//10)])
axes[0].set_xticklabels([str(monthly_counts.index[i]) for i in x[::max(1, len(x)//10)]], rotation=45)
axes[0].set_ylabel('Number of Ratings')
axes[0].set_title('Monthly Rating Volume')
axes[0].grid(axis='y', alpha=0.3)

axes[1].plot(x, monthly_avg.values, color='tomato', linewidth=2, marker='o', markersize=3)
axes[1].set_xticks(x[::max(1, len(x)//10)])
axes[1].set_xticklabels([str(monthly_avg.index[i]) for i in x[::max(1, len(x)//10)]], rotation=45)
axes[1].set_ylabel('Average Rating')
axes[1].set_title('Monthly Average Rating')
axes[1].set_ylim(1, 5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('step9_rating_trends.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: step9_rating_trends.png")
plt.close()

# ============================================================================
# STEP 10: JOIN INTEGRITY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: JOIN INTEGRITY CHECK")
print("=" * 80)

ratings_movie_ids = set(ratings['movieId'].unique())
movies_movie_ids  = set(movies['movieId'].unique())
ratings_user_ids  = set(ratings['userId'].unique())
users_user_ids    = set(users['userId'].unique())

print("\n1️⃣  RATINGS ↔ MOVIES (movieId)")
orphaned_movies = ratings_movie_ids - movies_movie_ids
unrated_movies  = movies_movie_ids  - ratings_movie_ids
print(f"   Movie IDs in ratings       : {len(ratings_movie_ids):,}")
print(f"   Movie IDs in catalog       : {len(movies_movie_ids):,}")
print(f"   In ratings but NOT catalog : {len(orphaned_movies):,}" +
      (" ⚠️" if orphaned_movies else " ✅"))
print(f"   In catalog but NOT rated   : {len(unrated_movies):,}")
print(f"   Join success rate          : "
      f"{len(ratings_movie_ids & movies_movie_ids)/len(ratings_movie_ids)*100:.2f}%")

print("\n2️⃣  RATINGS ↔ USERS (userId)")
orphaned_users  = ratings_user_ids - users_user_ids
inactive_users  = users_user_ids   - ratings_user_ids
print(f"   User IDs in ratings        : {len(ratings_user_ids):,}")
print(f"   User IDs in user file      : {len(users_user_ids):,}")
print(f"   In ratings but NOT users   : {len(orphaned_users):,}" +
      (" ⚠️" if orphaned_users else " ✅"))
print(f"   Users with no ratings      : {len(inactive_users):,}")
print(f"   Join success rate          : "
      f"{len(ratings_user_ids & users_user_ids)/len(ratings_user_ids)*100:.2f}%")

total_issues = len(orphaned_movies) + len(orphaned_users)

print("\n" + "=" * 80)
print("OVERALL DATA INTEGRITY SUMMARY")
print("=" * 80)

if total_issues == 0:
    print("✅ All joins are perfectly intact! No orphaned records found.")
else:
    print(f"⚠️  Found {total_issues} integrity issue(s):")
    if orphaned_movies:
        print(f"   - {len(orphaned_movies)} orphaned movie IDs in ratings")
    if orphaned_users:
        print(f"   - {len(orphaned_users)} orphaned user IDs in ratings")

# Bar chart — unique IDs per dataset
fig, ax = plt.subplots(figsize=(10, 5))
labels = ['Ratings\n(movieId)', 'Movies\ncatalog', 'Ratings\n(userId)', 'Users\nfile']
values = [len(ratings_movie_ids), len(movies_movie_ids),
          len(ratings_user_ids),  len(users_user_ids)]
colors = ['skyblue', 'lightgreen', 'coral', 'plum']
bars = ax.bar(labels, values, color=colors, edgecolor='black')
ax.set_ylabel('Number of Unique IDs')
ax.set_title('ID Distribution Across Datasets (Join Integrity)')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f'{val:,}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('step10_join_integrity.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: step10_join_integrity.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE! 🎉")
print("=" * 80)

print("\n📁 Generated Files:")
for f in [
    'step4_rating_distribution.png',
    'step5_user_activity.png',
    'step6_movie_popularity.png',
    'step7_genre_distribution.png',
    'step8_user_demographics.png',
    'step9_rating_trends.png',
    'step10_join_integrity.png',
]:
    print(f"   ✅ {f}")

print(f"""
💡 Key Insights:
   • {len(ratings):,} total ratings from {ratings['userId'].nunique():,} users
   • {len(movies):,} movies in catalog across {int((movies[GENRE_COLS] > 0).any().sum())} genres
   • {len(users):,} users | Age range {users['age'].min()}–{users['age'].max()} yrs
   • Average rating : {ratings['rating'].mean():.2f} / 5.0
   • Data quality   : {'Excellent ✅' if total_issues == 0 else f'{total_issues} issue(s) found ⚠️'}
""")
print("=" * 80)