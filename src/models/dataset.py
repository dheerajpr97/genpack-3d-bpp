import pandas as pd
from loguru import logger


class ProductDataset:
    """Load and preprocess product data from CSV files."""

    def __init__(self, file_path, seed=42):
        """Initialize the dataset."""
        self.file_path = file_path
        self.seed = seed
        self.data = self._load_data()

    def _load_data(self):
        """Load the dataset and normalize supported CSV formats."""
        try:
            df = pd.read_csv(
                self.file_path,
                dtype={
                    "id": "string",
                    "productid": "string",
                    "article": "string",
                    "order_id": "string",
                },
            )
            for column in ("id", "productid", "article", "order_id"):
                if column in df.columns:
                    df[column] = df[column].astype(str)
            logger.info(f"Loaded dataset from {self.file_path} with {len(df)} entries.")

            # Check if this is an orders_csv format file
            if "order_id" in df.columns and "id" in df.columns:
                logger.info("Detected orders_csv format. Converting to standard format...")

                # Map columns to standard format
                df_standard = pd.DataFrame(
                    {
                        "productid": df["id"],
                        "width": df["width"],
                        "length": df["length"],
                        "height": df["height"],
                        "weight": df["weight"],  # Ensure weight is included
                    }
                )

                # Add additional columns if available
                if "article" in df.columns:
                    df_standard["article"] = df["article"]
                if "product_group" in df.columns:
                    df_standard["product_group"] = df["product_group"]
                if "sequence" in df.columns:
                    df_standard["sequence"] = df["sequence"]
                if "order_id" in df.columns:
                    df_standard["order_id"] = df["order_id"]

                logger.info(
                    f"Converted to standard format with columns: {list(df_standard.columns)}"
                )
                return df_standard

            # Check if this is a SL (Solution Layer) format file
            elif "order" in df.columns and "productid" in df.columns:
                logger.info("Detected SL format. Converting to standard format...")

                # For SL format, use original dimensions and weight
                df_standard = pd.DataFrame(
                    {
                        "productid": df["productid"],
                        "width": df["width"],
                        "length": df["length"],
                        "height": df["height"],
                        "weight": df["weight"],  # Weight column from SL data
                    }
                )

                # Add additional columns if available
                if "order" in df.columns:
                    df_standard["order_id"] = df["order"]
                if "sequencenumber" in df.columns:
                    df_standard["sequence"] = df["sequencenumber"]

                logger.info(
                    f"Converted SL format to standard format with columns: {list(df_standard.columns)}"
                )
                return df_standard

            else:
                # Check if we have the basic required columns
                required_columns = ["productid", "width", "length", "height"]
                if all(col in df.columns for col in required_columns):
                    logger.info("Dataset appears to be in standard format.")

                    # Ensure weight column exists
                    if "weight" not in df.columns:
                        logger.warning(
                            "Weight column not found. Adding default weight of 1.0 for all items."
                        )
                        df["weight"] = 1.0

                    return df
                else:
                    logger.error(
                        f"Dataset format not recognized. Expected columns: {required_columns}"
                    )
                    logger.error(f"Found columns: {list(df.columns)}")
                    return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()

    def get_order(self, num_products):
        """Return a random subset of products."""
        if self.data.empty:
            logger.warning("Dataset is empty. Returning an empty order.")
            return pd.DataFrame()

        # Ensure we have the required columns
        required_columns = ["productid", "width", "length", "height", "weight"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()

        order = self.data.sample(n=min(num_products, len(self.data)), random_state=self.seed)
        logger.info(f"Generated order with {len(order)} items.")

        # Log weight information
        if "weight" in order.columns:
            total_weight = order["weight"].sum()
            avg_weight = order["weight"].mean()
            logger.info(
                f"Order weight statistics: Total={total_weight:.2f}, Average={avg_weight:.2f}"
            )

        return order

    def get_full_order(self):
        """Return the complete dataset as a single packing order."""
        if self.data.empty:
            logger.warning("Dataset is empty. Returning an empty order.")
            return pd.DataFrame()

        # Ensure we have the required columns
        required_columns = ["productid", "width", "length", "height", "weight"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()

        logger.info(f"Returning full order with {len(self.data)} items.")

        # Log weight information
        if "weight" in self.data.columns:
            total_weight = self.data["weight"].sum()
            avg_weight = self.data["weight"].mean()
            logger.info(
                f"Full order weight statistics: Total={total_weight:.2f}, Average={avg_weight:.2f}"
            )

        return self.data


# Example usage
if __name__ == "__main__":
    # Test with actual data files
    test_files = [
        "data/1.csv",
        "data/2.csv",
    ]

    for test_file in test_files:
        try:
            dataset = ProductDataset(test_file)
            if not dataset.data.empty:
                order = dataset.get_order(5)
                print(f"\nTest file: {test_file}")
                print(f"Columns: {list(order.columns)}")
                print(f"Sample data:")
                print(order.head(3))
                print(f"Weight column present: {'weight' in order.columns}")
                if "weight" in order.columns:
                    print(
                        f"Weight range: {order['weight'].min():.2f} - {order['weight'].max():.2f}"
                    )
        except Exception as e:
            print(f"Error testing {test_file}: {e}")
