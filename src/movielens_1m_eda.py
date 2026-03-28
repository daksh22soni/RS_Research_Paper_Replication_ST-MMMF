import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# STEP 1: LOAD AND UNDERSTAND EACH TABLE (STRUCTURE CHECK)
# ============================================================================

print("="*80)
print("STEP 1: STRUCTURE CHECK - Loading and understanding each table")
print("="*80)

# Load datasets with correct column names
users = pd.read_csv(
    "C:/Users/gahan/Documents/Daksh/sem 6/Recommendation System/project/ml-1m/users.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    header=None,
    names=["userId", "gender", "age", "occupation", "zip"]
)
movies = pd.read_csv(
    "C:/Users/gahan/Documents/Daksh/sem 6/Recommendation System/project/ml-1m/movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    header=None,
    names=["movieId", "title", "genres"]
)
ratings = pd.read_csv(
    "C:/Users/gahan/Documents/Daksh/sem 6/Recommendation System/project/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    header=None,
    names=["userId", "movieId", "rating", "timestamp"]
)

print("\n--- USERS.CSV ---")
print(f"Shape: {users.shape}")
print(f"Columns: {list(users.columns)}")
print(f"Data types:\n{users.dtypes}")
print(f"\nFirst 5 rows:")
print(users.head())

print("\n--- MOVIES.CSV ---")
print(f"Shape: {movies.shape}")
print(f"Columns: {list(movies.columns)}")
print(f"Data types:\n{movies.dtypes}")
print(f"\nFirst 5 rows:")
print(movies.head())

print("\n--- RATINGS.CSV ---")
print(f"Shape: {ratings.shape}")
print(f"Columns: {list(ratings.columns)}")
print(f"Data types:\n{ratings.dtypes}")
print(f"\nFirst 5 rows:")
print(ratings.head())

# ============================================================================
# STEP 2: SIZE OF DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 2: DATA SIZE ANALYSIS")
print("="*80)

num_users = users.shape[0]
num_movies = movies.shape[0]
num_ratings = ratings.shape[0]

print(f"\nTotal number of users: {num_users}")
print(f"Total number of movies: {num_movies}")
print(f"Total number of ratings: {num_ratings}")
print(f"\nData scale metrics:")
print(f"  - Sparsity: {(1 - num_ratings / (num_users * num_movies)) * 100:.4f}%")
print(f"  - Avg ratings per user: {num_ratings / num_users:.2f}")
print(f"  - Avg ratings per movie: {num_ratings / num_movies:.2f}")

# ============================================================================
# STEP 3: MISSING VALUES CHECK
# ============================================================================

print("\n" + "="*80)
print("STEP 3: MISSING VALUES CHECK")
print("="*80)

print("\n--- USERS TABLE ---")
missing_users = users.isnull().sum()
print(f"Missing values:\n{missing_users}")
if missing_users.sum() == 0:
    print("✓ No missing values in users table")

print("\n--- MOVIES TABLE ---")
missing_movies = movies.isnull().sum()
print(f"Missing values:\n{missing_movies}")
if missing_movies.sum() == 0:
    print("✓ No missing values in movies table")

print("\n--- RATINGS TABLE ---")
missing_ratings = ratings.isnull().sum()
print(f"Missing values:\n{missing_ratings}")
if missing_ratings.sum() == 0:
    print("✓ No missing values in ratings table")

# ============================================================================
# STEP 4: DUPLICATE RECORDS CHECK
# ============================================================================

print("\n" + "="*80)
print("STEP 4: DUPLICATE RECORDS CHECK")
print("="*80)

# Check for duplicate ratings (same user + same movie + same timestamp)
duplicate_ratings = ratings.duplicated(subset=['userId', 'movieId', 'timestamp']).sum()
print(f"\nDuplicate ratings (same user + movie + timestamp): {duplicate_ratings}")

# Check for same user rating same movie multiple times
user_movie_dupes = ratings.duplicated(subset=['userId', 'movieId'], keep=False).sum()
print(f"Ratings where same user rated same movie multiple times: {user_movie_dupes}")

if user_movie_dupes > 0:
    print(f"\nMovies rated multiple times by same user:")
    duplicate_user_movies = ratings[ratings.duplicated(subset=['userId', 'movieId'], keep=False)]
    print(duplicate_user_movies.groupby(['userId', 'movieId']).size().sort_values(ascending=False).head(10))

# Check for duplicate rows in users and movies
dup_users = users.duplicated().sum()
dup_movies = movies.duplicated().sum()
print(f"\nDuplicate rows in users table: {dup_users}")
print(f"Duplicate rows in movies table: {dup_movies}")

# ============================================================================
# STEP 5: RATING DISTRIBUTION (VERY IMPORTANT)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: RATING DISTRIBUTION ANALYSIS")
print("="*80)

rating_counts = ratings['rating'].value_counts().sort_index()
print(f"\nRating distribution:")
print(rating_counts)

print(f"\nPercentage distribution:")
rating_pct = (rating_counts / num_ratings * 100).round(2)
for rating, pct in rating_pct.items():
    print(f"  {rating} stars: {pct}% ({rating_counts[rating]} ratings)")

# Visualize rating distribution
plt.figure(figsize=(10, 6))
rating_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Rating Distribution - MovieLens 1M', fontsize=14, fontweight='bold')
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
for i, v in enumerate(rating_counts):
    plt.text(i, v + 1000, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('01_rating_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 01_rating_distribution.png")

print(f"\nRating statistics:")
print(f"  - Mean rating: {ratings['rating'].mean():.2f}")
print(f"  - Median rating: {ratings['rating'].median():.2f}")
print(f"  - Std Dev: {ratings['rating'].std():.2f}")
print(f"  - Min: {ratings['rating'].min()}, Max: {ratings['rating'].max()}")

# ============================================================================
# STEP 6: USER ACTIVITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: USER ACTIVITY ANALYSIS")
print("="*80)

user_ratings = ratings.groupby('userId').agg({
    'movieId': 'count',
    'rating': ['mean', 'std']
}).reset_index()
user_ratings.columns = ['userId', 'NumRatings', 'AvgRating', 'StdRating']

print(f"\nUser activity statistics:")
print(f"  - Average ratings per user: {user_ratings['NumRatings'].mean():.2f}")
print(f"  - Median ratings per user: {user_ratings['NumRatings'].median():.2f}")
print(f"  - Min ratings by a user: {user_ratings['NumRatings'].min()}")
print(f"  - Max ratings by a user: {user_ratings['NumRatings'].max()}")
print(f"  - Std Dev: {user_ratings['NumRatings'].std():.2f}")

# Very active users
top_users = user_ratings.nlargest(10, 'NumRatings')
print(f"\nTop 10 most active users:")
print(top_users[['userId', 'NumRatings', 'AvgRating']].to_string(index=False))

# Very inactive users
bottom_users = user_ratings.nsmallest(10, 'NumRatings')
print(f"\nTop 10 least active users:")
print(bottom_users[['userId', 'NumRatings', 'AvgRating']].to_string(index=False))

# Visualize user activity distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(user_ratings['NumRatings'], bins=50, color='steelblue', edgecolor='black')
axes[0].set_title('Distribution of Ratings Per User', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Ratings', fontsize=11)
axes[0].set_ylabel('Number of Users', fontsize=11)
axes[0].axvline(user_ratings['NumRatings'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {user_ratings["NumRatings"].mean():.2f}')
axes[0].axvline(user_ratings['NumRatings'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {user_ratings["NumRatings"].median():.2f}')
axes[0].legend()

# Box plot
axes[1].boxplot(user_ratings['NumRatings'], vert=True)
axes[1].set_title('Box Plot: Ratings Per User', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Ratings', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_user_activity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 02_user_activity_distribution.png")

# ============================================================================
# STEP 7: MOVIE POPULARITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 7: MOVIE POPULARITY ANALYSIS")
print("="*80)

movie_ratings = ratings.groupby('movieId').agg({
    'rating': ['count', 'mean', 'std']
}).reset_index()
movie_ratings.columns = ['movieId', 'NumRatings', 'AvgRating', 'StdRating']

# Merge with movie names
movie_ratings = movie_ratings.merge(movies[['movieId', 'title']], on='movieId', how='left')

print(f"\nMovie popularity statistics:")
print(f"  - Average ratings per movie: {movie_ratings['NumRatings'].mean():.2f}")
print(f"  - Median ratings per movie: {movie_ratings['NumRatings'].median():.2f}")
print(f"  - Min ratings for a movie: {movie_ratings['NumRatings'].min()}")
print(f"  - Max ratings for a movie: {movie_ratings['NumRatings'].max()}")
print(f"  - Std Dev: {movie_ratings['NumRatings'].std():.2f}")

# Most popular movies
print(f"\nTop 15 most popular movies (by number of ratings):")
top_movies = movie_ratings.nlargest(15, 'NumRatings')[['title', 'NumRatings', 'AvgRating']]
for idx, (_, row) in enumerate(top_movies.iterrows(), 1):
    print(f"  {idx:2d}. {row['title']:<50s} | Ratings: {row['NumRatings']:5d} | Avg: {row['AvgRating']:.2f}")

# Rarely rated movies
print(f"\nBottom 15 rarely rated movies:")
bottom_movies = movie_ratings.nsmallest(15, 'NumRatings')[['title', 'NumRatings', 'AvgRating']]
for idx, (_, row) in enumerate(bottom_movies.iterrows(), 1):
    print(f"  {idx:2d}. {row['title']:<50s} | Ratings: {row['NumRatings']:5d} | Avg: {row['AvgRating']:.2f}")

# Visualize movie popularity
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(movie_ratings['NumRatings'], bins=50, color='coral', edgecolor='black')
axes[0].set_title('Distribution of Ratings Per Movie', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Ratings', fontsize=11)
axes[0].set_ylabel('Number of Movies', fontsize=11)
axes[0].axvline(movie_ratings['NumRatings'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {movie_ratings["NumRatings"].mean():.2f}')
axes[0].axvline(movie_ratings['NumRatings'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {movie_ratings["NumRatings"].median():.2f}')
axes[0].legend()

# Top 15 movies bar chart
top_15 = movie_ratings.nlargest(15, 'NumRatings').sort_values('NumRatings')
axes[1].barh(range(len(top_15)), top_15['NumRatings'], color='coral', edgecolor='black')
axes[1].set_yticks(range(len(top_15)))
axes[1].set_yticklabels([title[:40] for title in top_15['title']], fontsize=9)
axes[1].set_title('Top 15 Most Popular Movies', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Ratings', fontsize=11)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('03_movie_popularity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 03_movie_popularity_distribution.png")

# ============================================================================
# STEP 8: GENRE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 8: GENRE ANALYSIS")
print("="*80)

# Extract genres - they're pipe-separated
genre_list = []
for genres in movies['genres']:
    if isinstance(genres, str):
        genre_list.extend(genres.split('|'))

genre_counts = Counter(genre_list)
genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count']).sort_values('Count', ascending=False)

print(f"\nTotal unique genres: {len(genre_counts)}")
print(f"\nGenre distribution:")
print(genre_df.to_string(index=False))

# Count movies per genre (note: movies can have multiple genres)
print(f"\nGenres by popularity:")
for idx, (genre, count) in enumerate(genre_df.values, 1):
    pct = (count / movies.shape[0]) * 100
    print(f"  {idx:2d}. {genre:<20s} | Count: {count:4d} | {pct:6.2f}%")

# Visualize genre distribution
plt.figure(figsize=(12, 6))
genre_df_top = genre_df.head(15)
plt.barh(range(len(genre_df_top)), genre_df_top['Count'], color='lightgreen', edgecolor='black')
plt.yticks(range(len(genre_df_top)), genre_df_top['Genre'])
plt.title('Top 15 Genres in MovieLens Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Number of Movies', fontsize=12)
plt.gca().invert_yaxis()
for i, v in enumerate(genre_df_top['Count']):
    plt.text(v + 10, i, str(v), va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('04_genre_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 04_genre_distribution.png")

# ============================================================================
# STEP 9: JOIN CHECK (RELATIONAL INTEGRITY)
# ============================================================================

print("\n" + "="*80)
print("STEP 9: JOIN CHECK - RELATIONAL INTEGRITY")
print("="*80)

# Check if all movies in ratings exist in movies table
missing_movies_in_ratings = ratings[~ratings['movieId'].isin(movies['movieId'])].shape[0]
print(f"\nRatings referencing non-existent movies: {missing_movies_in_ratings}")
if missing_movies_in_ratings == 0:
    print("✓ All movie IDs in ratings table are valid")

# Check if all users in ratings exist in users table
missing_users_in_ratings = ratings[~ratings['userId'].isin(users['userId'])].shape[0]
print(f"Ratings referencing non-existent users: {missing_users_in_ratings}")
if missing_users_in_ratings == 0:
    print("✓ All user IDs in ratings table are valid")

# Check if all users in ratings are in range
print(f"\nUser ID range in ratings: {ratings['userId'].min()} - {ratings['userId'].max()}")
print(f"User ID range in users table: {users['userId'].min()} - {users['userId'].max()}")

# Check if all movies in ratings are in range
print(f"Movie ID range in ratings: {ratings['movieId'].min()} - {ratings['movieId'].max()}")
print(f"Movie ID range in movies table: {movies['movieId'].min()} - {movies['movieId'].max()}")

print("\n✓ Relational integrity check complete")

# ============================================================================
# STEP 10: CORRELATION & BEHAVIORAL PATTERN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 10: CORRELATION & BEHAVIORAL PATTERN ANALYSIS")
print("="*80)

# Merge ratings with users and movies data
merged_data = ratings.merge(users, on='userId', how='left').merge(movies, on='movieId', how='left')

# 10.1: Age vs Number of Ratings
print("\n--- 10.1: AGE VS NUMBER OF RATINGS ---")
age_rating_activity = merged_data.groupby('age').agg({
    'rating': ['count', 'mean']
}).reset_index()
age_rating_activity.columns = ['Age', 'NumRatings', 'AvgRating']
print(age_rating_activity.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(age_rating_activity['Age'], age_rating_activity['NumRatings'], color='skyblue', edgecolor='black')
axes[0].set_title('Number of Ratings by Age Group', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Age', fontsize=11)
axes[0].set_ylabel('Total Ratings', fontsize=11)
axes[0].set_xticks(age_rating_activity['Age'])
axes[0].tick_params(axis='x', rotation=45)

axes[1].plot(age_rating_activity['Age'], age_rating_activity['AvgRating'], marker='o', linewidth=2, markersize=8, color='darkblue')
axes[1].set_title('Average Rating by Age Group', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Age', fontsize=11)
axes[1].set_ylabel('Average Rating', fontsize=11)
axes[1].set_xticks(age_rating_activity['Age'])
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_age_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 05_age_analysis.png")

# 10.2: Gender vs Average Rating
print("\n--- 10.2: GENDER VS AVERAGE RATING ---")
gender_analysis = merged_data.groupby('gender').agg({
    'rating': ['count', 'mean', 'std'],
    'userId': 'nunique'
}).reset_index()
gender_analysis.columns = ['Gender', 'TotalRatings', 'AvgRating', 'StdRating', 'UniqueUsers']
print(gender_analysis.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ratings count by gender
axes[0].bar(gender_analysis['Gender'], gender_analysis['TotalRatings'], color=['#FF69B4', '#4169E1'], edgecolor='black', width=0.6)
axes[0].set_title('Total Ratings by Gender', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Total Ratings', fontsize=11)
for i, v in enumerate(gender_analysis['TotalRatings']):
    axes[0].text(i, v + 10000, str(v), ha='center', fontweight='bold')

# Average rating by gender
axes[1].bar(gender_analysis['Gender'], gender_analysis['AvgRating'], color=['#FF69B4', '#4169E1'], edgecolor='black', width=0.6, alpha=0.7)
axes[1].set_title('Average Rating by Gender', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Average Rating', fontsize=11)
axes[1].set_ylim([3.0, 3.8])
for i, v in enumerate(gender_analysis['AvgRating']):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('06_gender_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 06_gender_analysis.png")

# 10.3: Occupation vs Average Rating
print("\n--- 10.3: OCCUPATION VS AVERAGE RATING ---")
occupation_analysis = merged_data.groupby('occupation').agg({
    'rating': ['count', 'mean'],
    'userId': 'nunique'
}).reset_index()
occupation_analysis.columns = ['Occupation', 'TotalRatings', 'AvgRating', 'UniqueUsers']
occupation_analysis = occupation_analysis.sort_values('TotalRatings', ascending=False)

print(occupation_analysis.head(15).to_string(index=False))

# Visualize top occupations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

top_occ = occupation_analysis.head(10).sort_values('TotalRatings')
axes[0].barh(range(len(top_occ)), top_occ['TotalRatings'], color='mediumpurple', edgecolor='black')
axes[0].set_yticks(range(len(top_occ)))
axes[0].set_yticklabels(top_occ['Occupation'], fontsize=10)
axes[0].set_title('Top 10 Occupations by Rating Count', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Total Ratings', fontsize=11)

axes[1].scatter(occupation_analysis['UniqueUsers'], occupation_analysis['AvgRating'], s=occupation_analysis['TotalRatings']/10, alpha=0.6, color='mediumpurple', edgecolors='black')
axes[1].set_title('Occupation: Users vs Average Rating (bubble size = total ratings)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Unique Users', fontsize=11)
axes[1].set_ylabel('Average Rating', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('07_occupation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 07_occupation_analysis.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE SUMMARY STATISTICS")
print("="*80)

summary = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      MOVIELENS 1M DATASET SUMMARY                          ║
╠══���═════════════════════════════════════════════════════════════════════════╣
║ DATASET SIZE                                                               ║
║  - Users:              {num_users:>8,}                                      ║
║  - Movies:             {num_movies:>8,}                                      ║
║  - Ratings:            {num_ratings:>8,}                                      ║
║  - Sparsity:           {(1 - num_ratings / (num_users * num_movies)) * 100:>7.3f}%                                     ║
╠═══════════════════════════════════════���════════════════════════════════════╣
║ RATINGS STATISTICS                                                         ║
║  - Mean Rating:        {ratings['rating'].mean():>8.2f}                                     ║
║  - Median Rating:      {ratings['rating'].median():>8.2f}                                     ║
║  - Std Dev:            {ratings['rating'].std():>8.2f}                                     ║
║  - Min/Max:            {ratings['rating'].min():.0f}/{ratings['rating'].max():.0f}                                       ║
╠════════════════════════════════════════════════════════════════════════════╣
║ USER BEHAVIOR                                                              ║
║  - Avg ratings/user:   {num_ratings / num_users:>8.2f}                                     ║
║  - Max ratings (user): {user_ratings['NumRatings'].max():>8,}                                      ║
║  - Min ratings (user): {user_ratings['NumRatings'].min():>8,}                                      ║
╠════════════════════════════════════════════════════════════════════════════╣
║ MOVIE STATISTICS                                                           ║
║  - Avg ratings/movie:  {num_ratings / num_movies:>8.2f}                                     ║
║  - Max ratings (movie):{movie_ratings['NumRatings'].max():>8,}                                      ║
║  - Min ratings (movie):{movie_ratings['NumRatings'].min():>8,}                                      ║
║  - Total genres:       {len(genre_counts):>8}                                       ║
╠════════════════════════════════════════════════════════════════════════════╣
║ DATA QUALITY                                                               ║
║  - Missing values:     ✓ None found                                        ║
║  - Duplicate ratings:  {duplicate_ratings}                                         ║
║  - Referential integrity: ✓ All valid                                      ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

print(summary)

# Save summary to file
with open('EDA_SUMMARY.txt', 'w',encoding="utf-8") as f:
    f.write(summary)
    f.write("\n\n" + "="*80 + "\n")
    f.write("DETAILED STATISTICS\n")
    f.write("="*80 + "\n\n")
    f.write("RATING DISTRIBUTION:\n")
    f.write(rating_counts.to_string())
    f.write("\n\n" + "="*80 + "\n")
    f.write("TOP 20 MOST POPULAR MOVIES:\n")
    f.write("="*80 + "\n")
    f.write(movie_ratings.nlargest(20, 'NumRatings')[['title', 'NumRatings', 'AvgRating']].to_string())
    f.write("\n\n" + "="*80 + "\n")
    f.write("TOP 20 MOST ACTIVE USERS:\n")
    f.write("="*80 + "\n")
    f.write(user_ratings.nlargest(20, 'NumRatings')[['userId', 'NumRatings', 'AvgRating']].to_string())
    f.write("\n\n" + "="*80 + "\n")
    f.write("GENRE DISTRIBUTION:\n")
    f.write("="*80 + "\n")
    f.write(genre_df.to_string())

print("\n✓ Saved: EDA_SUMMARY.txt")

print("\n" + "="*80)
print("EDA COMPLETE! Generated visualizations:")
print("="*80)
print("  1. 01_rating_distribution.png")
print("  2. 02_user_activity_distribution.png")
print("  3. 03_movie_popularity_distribution.png")
print("  4. 04_genre_distribution.png")
print("  5. 05_age_analysis.png")
print("  6. 06_gender_analysis.png")
print("  7. 07_occupation_analysis.png")
print("  + EDA_SUMMARY.txt")
print("="*80)