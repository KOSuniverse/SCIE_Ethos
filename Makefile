# SCIE Ethos Makefile

.PHONY: dp-live-audit test-dp clean help

# Data Processing Live Audit
dp-live-audit:
	@echo "🚀 Running Data Processing Live Audit..."
	python scripts/run_live_dp_validation.py

# Test enhanced DP implementation
test-dp:
	@echo "🧪 Testing Enhanced DP Implementation..."
	python scripts/test_enhanced_dp.py

# Clean generated artifacts
clean:
	@echo "🧹 Cleaning generated artifacts..."
	@rm -rf /Project_Root/04_Data/03_Summaries/DP_LIVE/ || true
	@rm -rf audit/ || true

# Help
help:
	@echo "Available targets:"
	@echo "  dp-live-audit  - Run autonomous DP live validation"
	@echo "  test-dp        - Test enhanced DP implementation"
	@echo "  clean          - Clean generated artifacts"
	@echo "  help           - Show this help"
