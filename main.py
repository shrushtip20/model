from recommender import SkinProductRecommender

def main():
    # Create an instance of the recommender
    recommender = SkinProductRecommender()
    
    # Test with a skin condition
    condition = "ringworm"
    print(f"\nGetting recommendations for: {condition}")
    print("-" * 50)
    
    # Get recommendations
    recommendations, message = recommender.get_recommendations(
        skin_condition=condition,
        num_recommendations=3,
        max_price=30
    )
    
    print(f"Message: {message}")
    print("\nRecommended Products:")
    print(recommendations)

if __name__ == "__main__":
    main()