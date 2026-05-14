# Configuration for the application
MODEL_NAME = "arn:aws:bedrock:us-east-1:651818016290:application-inference-profile/kzx60kroqtkq"
AWS_REGION = "us-east-1"
TEMPERATURE = 1.0
MAX_TOKENS = 4000

USE_NAVIGATOR = False

# Bedrock pricing for the configured model, used by the TUI cost tally.
# Update if MODEL_NAME points at something other than Claude Sonnet 4.5.
PRICE_INPUT_PER_MTOK = 3.00
PRICE_OUTPUT_PER_MTOK = 15.00
PRICE_CACHE_READ_PER_MTOK = 0.30
PRICE_CACHE_WRITE_PER_MTOK = 3.75