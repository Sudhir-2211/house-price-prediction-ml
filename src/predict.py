"""Command-Line Interface for House Price Inference.

Author: Sudhir Kumar (GitHub: Sudhir-2211)
"""

import os
import argparse
import joblib
import pandas as pd


def predict_price(
    sqft_living: int,
    bedrooms: int,
    bathrooms: float,
    location_tier: str,
    year_built: int,
    lot_size_acres: float,
    garage_spaces: int,
    condition_rating: int,
    has_pool: int,
    model_path: str = None,
) -> dict:
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "house_price_pipeline.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run 'python -m src.train' first.")

    pipeline = joblib.load(model_path)

    sample = pd.DataFrame([{
        "sqft_living": sqft_living,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location_tier": location_tier,
        "year_built": year_built,
        "lot_size_acres": lot_size_acres,
        "garage_spaces": garage_spaces,
        "condition_rating": condition_rating,
        "has_pool": has_pool,
    }])

    pred = float(pipeline.predict(sample)[0])
    # Statistical error band (~90% interval based on empirical RMSE ~$50,000)
    lower = max(50000.0, pred - 1.645 * 50000.0)
    upper = pred + 1.645 * 50000.0

    return {
        "estimated_price": round(pred, 2),
        "confidence_lower": round(lower, 2),
        "confidence_upper": round(upper, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="House Price Regression Predictor")
    parser.add_argument("--sqft", type=int, default=2400, help="Square footage of living area")
    parser.add_argument("--beds", type=int, default=3, help="Number of bedrooms")
    parser.add_argument("--baths", type=float, default=2.5, help="Number of bathrooms")
    parser.add_argument("--location", default="Suburban Central", choices=["Urban Prime", "Suburban Central", "Outer Suburban", "Rural Scenic"])
    parser.add_argument("--year", type=int, default=2005, help="Year built")
    parser.add_argument("--lot", type=float, default=0.25, help="Lot size in acres")
    parser.add_argument("--garage", type=int, default=2, help="Garage car spaces")
    parser.add_argument("--condition", type=int, default=4, choices=[1, 2, 3, 4, 5], help="Condition rating (1-5)")
    parser.add_argument("--pool", type=int, default=0, choices=[0, 1], help="Has swimming pool (0 or 1)")

    args = parser.parse_args()

    result = predict_price(
        sqft_living=args.sqft,
        bedrooms=args.beds,
        bathrooms=args.baths,
        location_tier=args.location,
        year_built=args.year,
        lot_size_acres=args.lot,
        garage_spaces=args.garage,
        condition_rating=args.condition,
        has_pool=args.pool,
    )

    print("\n" + "=" * 60)
    print("           HOUSE VALUATION ESTIMATION RESULT")
    print("=" * 60)
    print(f"Property Specifications:")
    print(f"  • Size        : {args.sqft} sq ft | {args.beds} Beds | {args.baths} Baths")
    print(f"  • Location    : {args.location} | Built: {args.year}")
    print(f"  • Lot / Garage: {args.lot} Acres | {args.garage} Car Garage | Pool: {'Yes' if args.pool else 'No'}")
    print("-" * 60)
    print(f"Estimated Market Value  : ${result['estimated_price']:,.2f}")
    print(f"Estimated 90% Band     : ${result['confidence_lower']:,.2f} - ${result['confidence_upper']:,.2f}")
    print("=" * 60)
    print("DISCLAIMER: Predictions are statistical estimates for educational")
    print("demonstration only, not official professional appraisals.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
