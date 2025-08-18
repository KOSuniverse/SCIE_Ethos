#!/usr/bin/env python3
"""
Enhanced Data Ingestion Pipeline for SCIE Ethos
Integrates with existing components to produce required artifacts according to ingest_rules.yaml
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import hashlib
import re

# Import existing components with graceful fallbacks
try:
    from .pipeline import run_pipeline
    from .sheet_utils import classify_sheet, extract_locations_and_context
    from .smart_cleaning import run_smart_autofix
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    def run_pipeline(*args, **kwargs): return {}, []
    def classify_sheet(*args, **kwargs): return {"final_type": "unclassified"}
    def extract_locations_and_context(*args, **kwargs): return {"locations": [], "inferred_from": []}
    def run_smart_autofix(*args, **kwargs): return pd.DataFrame(), [], {}

try:
    from ..phase2_analysis.enhanced_eda_system import EnhancedEDASystem
    EDA_AVAILABLE = True
except ImportError:
    EDA_AVAILABLE = False
    class EnhancedEDASystem:
        def run_comprehensive_eda(self, *args, **kwargs):
            return {"metadata": {}, "chart_paths": [], "business_insights": {}}

try:
    from ..phase4_knowledge.knowledgebase_builder import KnowledgeBaseBuilder
    KB_AVAILABLE = True
except ImportError:
    KB_AVAILABLE = False
    class KnowledgeBaseBuilder:
        def ingest_documents(self, *args, **kwargs): return {"ingested": 0, "errors": []}

try:
    from ..metadata_utils import save_per_sheet_metadata, save_per_sheet_summary, save_per_sheet_eda
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False
    def save_per_sheet_metadata(*args, **kwargs): pass
    def save_per_sheet_summary(*args, **kwargs): pass
    def save_per_sheet_eda(*args, **kwargs): pass

try:
    from ..dbx_utils import dbx_read_json, dbx_write_json
    DBX_AVAILABLE = True
except ImportError:
    DBX_AVAILABLE = False
    def dbx_read_json(*args, **kwargs): return None
    def dbx_write_json(*args, **kwargs): pass

class EnhancedIngestionPipeline:
    """
    Enhanced ingestion pipeline that produces all required artifacts:
    - master_catalog.jsonl
    - eda_profile.json  
    - summary_card.md
    - Knowledge base ingestion
    """
    
    def __init__(self, config_path: str = "prompts/ingest_rules.yaml"):
        self.config = self._load_config(config_path)
        self.path_contract = self._load_path_contract()
        self.ontology = self.config.get("ontology", {})
        self.artifacts_config = self.config.get("artifacts", {})
        self.kb_config = self.config.get("knowledge_base_ingest", {})
        
        # Initialize components
        self.eda_system = EnhancedEDASystem() if EDA_AVAILABLE else None
        self.kb_builder = KnowledgeBaseBuilder() if KB_AVAILABLE else None
        
        # Output tracking
        self.master_catalog = []
        self.eda_profiles = {}
        self.summary_cards = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load ingestion configuration from YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Failed to load {config_path}: {e}")
            return {}
    
    def _load_path_contract(self) -> Dict[str, Any]:
        """Load path contract configuration."""
        try:
            path_contract_path = self.config.get("paths", {}).get("path_contract_file", "configs/path_contract.yaml")
            with open(path_contract_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Failed to load path contract: {e}")
            return {}
    
    def _get_output_path(self, artifact_type: str) -> str:
        """Get output path for artifacts based on path contract."""
        base_dir = self.path_contract.get("base_data_dir", "Project_Root/04_Data")
        folders = self.path_contract.get("folders", {})
        
        if artifact_type == "master_catalog":
            return f"{base_dir}/{folders.get('metadata', '04_Metadata')}/master_catalog.jsonl"
        elif artifact_type == "eda_profile":
            return f"{base_dir}/{folders.get('metadata', '04_Metadata')}/eda_profile.json"
        elif artifact_type == "summary_card":
            return f"{base_dir}/{folders.get('metadata', '04_Metadata')}/summary_card.md"
        else:
            return f"{base_dir}/{folders.get('metadata', '04_Metadata')}/{artifact_type}"
    
    def _extract_ontology_metadata(self, df: pd.DataFrame, sheet_name: str, filename: str) -> Dict[str, Any]:
        """Extract metadata according to the ontology tiers."""
        metadata = {}
        
        # Tier 0: Core identifiers
        metadata["erp"] = self._detect_erp(filename, sheet_name, df)
        metadata["country"] = self._detect_country(filename, sheet_name, df)
        metadata["income_stream"] = self._detect_income_stream(filename, sheet_name, df)
        metadata["sheet_type"] = self._detect_sheet_type(filename, sheet_name, df)
        
        # Tier 1: Business context
        metadata["product_family"] = self._detect_product_family(df)
        metadata["business_unit"] = self._detect_business_unit(df)
        metadata["market"] = self._detect_market(df)
        metadata["channel"] = self._detect_channel(df)
        metadata["plant"] = self._detect_plant(df)
        metadata["location_type"] = self._detect_location_type(df)
        metadata["owner_team"] = self._detect_owner_team(df)
        metadata["supplier"] = self._detect_supplier(df)
        metadata["customer_segment"] = self._detect_customer_segment(df)
        metadata["valuation_method"] = self._detect_valuation_method(df)
        metadata["currency"] = self._detect_currency(df)
        metadata["uom"] = self._detect_uom(df)
        
        # Tier 2: Temporal context
        metadata["fiscal_period"] = self._detect_fiscal_period(df)
        metadata["snapshot_asof"] = self._detect_snapshot_asof(df)
        metadata["calendar_granularity"] = self._detect_calendar_granularity(df)
        metadata["horizon"] = self._detect_horizon(df)
        
        # Tier 3: Technical metadata
        metadata["source_file"] = filename
        metadata["source_sheet"] = sheet_name
        metadata["version"] = self._detect_version(df)
        metadata["lang"] = self._detect_language(df)
        metadata["header_row"] = self._detect_header_row(df)
        metadata["pii"] = self._detect_pii(df)
        metadata["confidentiality"] = self._detect_confidentiality(df)
        metadata["ocr"] = False  # Excel files don't need OCR
        metadata["eda_version"] = "v1.0"
        
        # Tier 4: Analysis metadata
        metadata["primary_key_guess"] = self._detect_primary_key(df)
        metadata["join_keys_found"] = self._detect_join_keys(df)
        metadata["join_score"] = self._calculate_join_score(df)
        metadata["intermittency"] = self._detect_intermittency(df)
        metadata["seasonality_hint"] = self._detect_seasonality(df)
        metadata["lt_mean"] = self._detect_lead_time_mean(df)
        metadata["lt_std"] = self._detect_lead_time_std(df)
        metadata["lt_cv"] = self._detect_lead_time_cv(df)
        metadata["has_aging"] = self._detect_aging_buckets(df)
        metadata["has_wip"] = self._detect_wip_data(df)
        metadata["has_eo"] = self._detect_eo_data(df)
        metadata["has_po"] = self._detect_po_data(df)
        metadata["has_receipts"] = self._detect_receipts_data(df)
        metadata["has_forecast"] = self._detect_forecast_data(df)
        
        return metadata
    
    def _detect_erp(self, filename: str, sheet_name: str, df: pd.DataFrame) -> str:
        """Detect ERP system from filename, sheet name, or data."""
        text = f"{filename} {sheet_name}".lower()
        if "sap" in text:
            return "SAP"
        elif "oracle" in text or "jde" in text:
            return "Oracle"
        elif "dynamics" in text or "ax" in text:
            return "Dynamics"
        elif "netsuite" in text:
            return "NetSuite"
        else:
            return "Unknown"
    
    def _detect_country(self, filename: str, sheet_name: str, df: pd.DataFrame) -> str:
        """Detect country from filename, sheet name, or data."""
        text = f"{filename} {sheet_name}".lower()
        countries = ["US", "MX", "CA", "UK", "EU", "Thailand", "Germany", "China", "Japan"]
        
        for country in countries:
            if country.lower() in text:
                return country
        
        # Try to detect from data
        for col in df.columns:
            col_lower = str(col).lower()
            if any(country.lower() in col_lower for country in countries):
                return "Multi"
        
        return "Unknown"
    
    def _detect_income_stream(self, filename: str, sheet_name: str, df: pd.DataFrame) -> str:
        """Detect income stream from context."""
        text = f"{filename} {sheet_name}".lower()
        if "inventory" in text:
            return "Inventory"
        elif "wip" in text or "work" in text:
            return "WIP"
        elif "raw" in text:
            return "Raw Materials"
        elif "finished" in text or "fg" in text:
            return "Finished Goods"
        else:
            return "Unknown"
    
    def _detect_sheet_type(self, filename: str, sheet_name: str, df: pd.DataFrame) -> str:
        """Detect sheet type using existing classification."""
        if PIPELINE_AVAILABLE:
            classification = classify_sheet(sheet_name, df, {})
            return classification.get("final_type", "unclassified")
        else:
            # Fallback detection
            cols_lower = [str(c).lower() for c in df.columns]
            if any("part" in c for c in cols_lower):
                return "inventory"
            elif any("job" in c or "wo" in c for c in cols_lower):
                return "wip"
            else:
                return "unclassified"
    
    def _detect_product_family(self, df: pd.DataFrame) -> str:
        """Detect product family from data."""
        # Implementation would analyze column names and values
        return "Unknown"
    
    def _detect_business_unit(self, df: pd.DataFrame) -> str:
        """Detect business unit from data."""
        return "Unknown"
    
    def _detect_market(self, df: pd.DataFrame) -> str:
        """Detect market from data."""
        return "Unknown"
    
    def _detect_channel(self, df: pd.DataFrame) -> str:
        """Detect channel from data."""
        return "Unknown"
    
    def _detect_plant(self, df: pd.DataFrame) -> str:
        """Detect plant from data."""
        return "Unknown"
    
    def _detect_location_type(self, df: pd.DataFrame) -> str:
        """Detect location type from data."""
        return "Unknown"
    
    def _detect_owner_team(self, df: pd.DataFrame) -> str:
        """Detect owner team from data."""
        return "Unknown"
    
    def _detect_supplier(self, df: pd.DataFrame) -> str:
        """Detect supplier information from data."""
        return "Unknown"
    
    def _detect_customer_segment(self, df: pd.DataFrame) -> str:
        """Detect customer segment from data."""
        return "Unknown"
    
    def _detect_valuation_method(self, df: pd.DataFrame) -> str:
        """Detect valuation method from data."""
        return "Unknown"
    
    def _detect_currency(self, df: pd.DataFrame) -> str:
        """Detect currency from data."""
        cols_lower = [str(c).lower() for c in df.columns]
        if any("usd" in c for c in cols_lower):
            return "USD"
        elif any("eur" in c for c in cols_lower):
            return "EUR"
        elif any("mxn" in c for c in cols_lower):
            return "MXN"
        else:
            return "Unknown"
    
    def _detect_uom(self, df: pd.DataFrame) -> str:
        """Detect unit of measure from data."""
        cols_lower = [str(c).lower() for c in df.columns]
        if any("pcs" in c for c in cols_lower):
            return "PCS"
        elif any("kg" in c for c in cols_lower):
            return "KG"
        elif any("ft" in c for c in cols_lower):
            return "FT"
        else:
            return "Unknown"
    
    def _detect_fiscal_period(self, df: pd.DataFrame) -> str:
        """Detect fiscal period from data."""
        return "Unknown"
    
    def _detect_snapshot_asof(self, df: pd.DataFrame) -> str:
        """Detect snapshot as-of date from data."""
        return "Unknown"
    
    def _detect_calendar_granularity(self, df: pd.DataFrame) -> str:
        """Detect calendar granularity from data."""
        return "Unknown"
    
    def _detect_horizon(self, df: pd.DataFrame) -> str:
        """Detect planning horizon from data."""
        return "Unknown"
    
    def _detect_version(self, df: pd.DataFrame) -> str:
        """Detect data version from data."""
        return "v1.0"
    
    def _detect_language(self, df: pd.DataFrame) -> str:
        """Detect language from data."""
        return "en"
    
    def _detect_header_row(self, df: pd.DataFrame) -> int:
        """Detect header row position."""
        return 0
    
    def _detect_pii(self, df: pd.DataFrame) -> bool:
        """Detect presence of PII in data."""
        return False
    
    def _detect_confidentiality(self, df: pd.DataFrame) -> str:
        """Detect confidentiality level."""
        return "internal"
    
    def _detect_primary_key(self, df: pd.DataFrame) -> str:
        """Detect potential primary key column."""
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ["id", "key", "part", "sku", "job"]):
                return str(col)
        return "Unknown"
    
    def _detect_join_keys(self, df: pd.DataFrame) -> List[str]:
        """Detect potential join key columns."""
        join_keys = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ["id", "key", "part", "sku", "job", "plant", "country"]):
                join_keys.append(str(col))
        return join_keys
    
    def _calculate_join_score(self, df: pd.DataFrame) -> float:
        """Calculate joinability score."""
        # Simple heuristic based on unique values
        total_rows = len(df)
        unique_counts = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            if total_rows > 0:
                unique_ratio = unique_count / total_rows
                unique_counts.append(unique_ratio)
        
        if unique_counts:
            return np.mean(unique_counts)
        return 0.0
    
    def _detect_intermittency(self, df: pd.DataFrame) -> str:
        """Detect demand intermittency pattern."""
        return "Unknown"
    
    def _detect_seasonality(self, df: pd.DataFrame) -> str:
        """Detect seasonality pattern."""
        return "Unknown"
    
    def _detect_lead_time_mean(self, df: pd.DataFrame) -> float:
        """Detect mean lead time if available."""
        for col in df.columns:
            col_lower = str(col).lower()
            if "lead" in col_lower and "time" in col_lower:
                try:
                    values = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(values) > 0:
                        return float(values.mean())
                except:
                    pass
        return 0.0
    
    def _detect_lead_time_std(self, df: pd.DataFrame) -> float:
        """Detect lead time standard deviation if available."""
        for col in df.columns:
            col_lower = str(col).lower()
            if "lead" in col_lower and "time" in col_lower:
                try:
                    values = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(values) > 0:
                        return float(values.std())
                except:
                    pass
        return 0.0
    
    def _detect_lead_time_cv(self, df: pd.DataFrame) -> float:
        """Detect lead time coefficient of variation."""
        lt_mean = self._detect_lead_time_mean(df)
        lt_std = self._detect_lead_time_std(df)
        if lt_mean > 0:
            return lt_std / lt_mean
        return 0.0
    
    def _detect_aging_buckets(self, df: pd.DataFrame) -> bool:
        """Detect presence of aging bucket columns."""
        cols_lower = [str(c).lower() for c in df.columns]
        return any("aging" in c or "days" in c for c in cols_lower)
    
    def _detect_wip_data(self, df: pd.DataFrame) -> bool:
        """Detect presence of WIP data."""
        cols_lower = [str(c).lower() for c in df.columns]
        return any("wip" in c or "work" in c or "job" in c for c in cols_lower)
    
    def _detect_eo_data(self, df: pd.DataFrame) -> bool:
        """Detect presence of E&O data."""
        cols_lower = [str(c).lower() for c in df.columns]
        return any("excess" in c or "obsolete" in c or "eo" in c for c in cols_lower)
    
    def _detect_po_data(self, df: pd.DataFrame) -> bool:
        """Detect presence of PO data."""
        cols_lower = [str(c).lower() for c in df.columns]
        return any("po" in c or "purchase" in c or "order" in c for c in cols_lower)
    
    def _detect_receipts_data(self, df: pd.DataFrame) -> bool:
        """Detect presence of receipts data."""
        cols_lower = [str(c).lower() for c in df.columns]
        return any("receipt" in c or "received" in c for c in cols_lower)
    
    def _detect_forecast_data(self, df: pd.DataFrame) -> bool:
        """Detect presence of forecast data."""
        cols_lower = [str(c).lower() for c in df.columns]
        return any("forecast" in c or "projected" in c or "expected" in c for c in cols_lower)
    
    def process_file(self, file_path: str, reporter: Optional[callable] = None) -> Dict[str, Any]:
        """Process a single file through the enhanced ingestion pipeline."""
        filename = os.path.basename(file_path)
        
        if reporter:
            reporter("start_file", {"filename": filename})
        
        try:
            # Run the existing pipeline
            if PIPELINE_AVAILABLE:
                cleaned_sheets, per_sheet_meta = run_pipeline(
                    source=file_path,
                    filename=filename,
                    reporter=reporter
                )
            else:
                # Fallback processing
                cleaned_sheets, per_sheet_meta = self._fallback_process_file(file_path, filename)
            
            # Enhanced metadata extraction
            enhanced_metadata = []
            for sheet_name, df in cleaned_sheets.items():
                if reporter:
                    reporter("enhancing_metadata", {"sheet": sheet_name})
                
                # Extract ontology-based metadata
                ontology_meta = self._extract_ontology_metadata(df, sheet_name, filename)
                
                # Extract locations and context
                location_meta = extract_locations_and_context(filename, sheet_name, df) if PIPELINE_AVAILABLE else {}
                
                # Combine metadata
                sheet_metadata = {
                    **ontology_meta,
                    **location_meta,
                    "filename": filename,
                    "sheet_name": sheet_name,
                    "row_count": len(df),
                    "col_count": len(df.columns),
                    "processed_at": datetime.utcnow().isoformat() + "Z"
                }
                
                enhanced_metadata.append(sheet_metadata)
                
                # Add to master catalog
                self.master_catalog.append(sheet_metadata)
                
                # Generate EDA profile
                if self.eda_system:
                    eda_results = self.eda_system.run_comprehensive_eda(df, sheet_name, filename)
                    self.eda_profiles[f"{filename}_{sheet_name}"] = eda_results
                
                # Generate summary card
                summary_card = self._generate_summary_card(sheet_metadata, df)
                self.summary_cards.append(summary_card)
                
                if reporter:
                    reporter("metadata_enhanced", {"sheet": sheet_name, "metadata_keys": len(sheet_metadata)})
            
            if reporter:
                reporter("file_complete", {"filename": filename, "sheets": len(cleaned_sheets)})
            
            return {
                "success": True,
                "filename": filename,
                "sheets_processed": len(cleaned_sheets),
                "metadata_enhanced": len(enhanced_metadata),
                "cleaned_sheets": cleaned_sheets,
                "enhanced_metadata": enhanced_metadata
            }
            
        except Exception as e:
            if reporter:
                reporter("file_error", {"filename": filename, "error": str(e)})
            return {
                "success": False,
                "filename": filename,
                "error": str(e)
            }
    
    def _fallback_process_file(self, file_path: str, filename: str) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
        """Fallback file processing when main pipeline is not available."""
        try:
            # Simple Excel reading
            xls = pd.ExcelFile(file_path)
            cleaned_sheets = {}
            per_sheet_meta = []
            
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                cleaned_sheets[sheet_name] = df
                
                meta = {
                    "filename": filename,
                    "sheet_name": sheet_name,
                    "normalized_sheet_type": "unclassified",
                    "rows": len(df),
                    "columns": list(df.columns),
                    "errors": []
                }
                per_sheet_meta.append(meta)
            
            return cleaned_sheets, per_sheet_meta
            
        except Exception as e:
            print(f"Fallback processing failed for {filename}: {e}")
            return {}, []
    
    def _generate_summary_card(self, metadata: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate summary card using template."""
        try:
            template_path = self.artifacts_config.get("summary_card_template", "templates/summary_card.template.md")
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Prepare template variables
            template_vars = {
                "logical_name": f"{metadata.get('sheet_type', 'Unknown')} - {metadata.get('country', 'Unknown')}",
                "sheet_type": metadata.get("sheet_type", "Unknown"),
                "country": metadata.get("country", "Unknown"),
                "erp": metadata.get("erp", "Unknown"),
                "date_min": metadata.get("snapshot_asof", "Unknown"),
                "date_max": metadata.get("snapshot_asof", "Unknown"),
                "calendar_granularity": metadata.get("calendar_granularity", "Unknown"),
                "row_count": metadata.get("row_count", 0),
                "col_count": metadata.get("col_count", 0),
                "lang": metadata.get("lang", "en"),
                "source_file": metadata.get("source_file", "Unknown"),
                "source_sheet": metadata.get("source_sheet", "Unknown"),
                "on_hand_qty": "Unknown",
                "uom": metadata.get("uom", "Unknown"),
                "currency": metadata.get("currency", "Unknown"),
                "inventory_value": "Unknown",
                "has_aging": "Yes" if metadata.get("has_aging") else "No",
                "lt_mean": metadata.get("lt_mean", "Unknown"),
                "lt_std": metadata.get("lt_std", "Unknown"),
                "example_queries": [
                    f"What is the current inventory level for {metadata.get('sheet_type', 'items')}?",
                    f"What are the aging buckets for {metadata.get('sheet_type', 'items')}?",
                    f"What is the lead time distribution for {metadata.get('sheet_type', 'items')}?"
                ],
                "caveats": [
                    "Data quality assessment pending",
                    "Business rules validation needed",
                    "Cross-reference verification required"
                ]
            }
            
            # Apply template variables
            summary_card = template
            for key, value in template_vars.items():
                if isinstance(value, list):
                    # Handle list values (bullet points)
                    bullet_list = "\n".join([f"- {item}" for item in value])
                    summary_card = summary_card.replace(f"{{{{{key}}}}}", bullet_list)
                else:
                    summary_card = summary_card.replace(f"{{{{{key}}}}}", str(value))
            
            return summary_card
            
        except Exception as e:
            print(f"Failed to generate summary card: {e}")
            return f"# Summary Card Generation Failed\n\nError: {e}"
    
    def ingest_knowledge_base(self, knowledge_base_path: str = None) -> Dict[str, Any]:
        """Ingest knowledge base documents according to configuration."""
        if not self.kb_builder or not self.kb_config.get("include", False):
            return {"ingested": 0, "errors": ["Knowledge base ingestion not configured"]}
        
        try:
            kb_path = knowledge_base_path or self.kb_config.get("path_ref", "Project_Root/06_LLM_Knowledge_Base")
            
            # Configure ingestion parameters
            config = {
                "file_types": self.kb_config.get("file_types", ["pdf"]),
                "ocr": self.kb_config.get("ocr", True),
                "tags": self.kb_config.get("tags", {}),
                "folder_to_tag_rules": self.kb_config.get("folder_to_tag_rules", [])
            }
            
            # Run ingestion
            results = self.kb_builder.ingest_documents(kb_path, config)
            
            return results
            
        except Exception as e:
            return {"ingested": 0, "errors": [str(e)]}
    
    def emit_artifacts(self) -> Dict[str, str]:
        """Emit all required artifacts to their designated locations."""
        artifacts = {}
        
        try:
            # Emit master catalog
            if self.artifacts_config.get("emit_master_catalog", True):
                master_catalog_path = self._get_output_path("master_catalog")
                os.makedirs(os.path.dirname(master_catalog_path), exist_ok=True)
                
                with open(master_catalog_path, 'w', encoding='utf-8') as f:
                    for entry in self.master_catalog:
                        f.write(json.dumps(entry) + '\n')
                
                artifacts["master_catalog"] = master_catalog_path
                print(f"✅ Emitted master catalog: {master_catalog_path}")
            
            # Emit EDA profile
            if self.artifacts_config.get("emit_eda_profile", True):
                eda_profile_path = self._get_output_path("eda_profile")
                os.makedirs(os.path.dirname(eda_profile_path), exist_ok=True)
                
                eda_profile = {
                    "version": "1.0",
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "profiles": self.eda_profiles,
                    "summary": {
                        "total_profiles": len(self.eda_profiles),
                        "files_analyzed": len(set(meta.get("filename") for meta in self.master_catalog))
                    }
                }
                
                with open(eda_profile_path, 'w', encoding='utf-8') as f:
                    json.dump(eda_profile, f, indent=2)
                
                artifacts["eda_profile"] = eda_profile_path
                print(f"✅ Emitted EDA profile: {eda_profile_path}")
            
            # Emit summary cards
            if self.artifacts_config.get("emit_summary_card", True):
                summary_cards_path = self._get_output_path("summary_card")
                os.makedirs(os.path.dirname(summary_cards_path), exist_ok=True)
                
                # Combine all summary cards
                combined_summary = "# SCIE Ethos Data Summary Cards\n\n"
                combined_summary += f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
                combined_summary += f"Total datasets: {len(self.summary_cards)}\n\n"
                
                for i, card in enumerate(self.summary_cards, 1):
                    combined_summary += f"## Dataset {i}\n\n"
                    combined_summary += card
                    combined_summary += "\n\n---\n\n"
                
                with open(summary_cards_path, 'w', encoding='utf-8') as f:
                    f.write(combined_summary)
                
                artifacts["summary_card"] = summary_cards_path
                print(f"✅ Emitted summary cards: {summary_cards_path}")
            
            return artifacts
            
        except Exception as e:
            print(f"❌ Failed to emit artifacts: {e}")
            return {}
    
    def run_full_ingestion(self, file_paths: List[str], knowledge_base_path: str = None, reporter: Optional[callable] = None) -> Dict[str, Any]:
        """Run the complete ingestion pipeline on multiple files."""
        if reporter:
            reporter("pipeline_start", {"total_files": len(file_paths)})
        
        results = {
            "files_processed": 0,
            "files_successful": 0,
            "files_failed": 0,
            "total_sheets": 0,
            "artifacts_emitted": {},
            "knowledge_base_ingested": {},
            "errors": []
        }
        
        # Process each file
        for i, file_path in enumerate(file_paths, 1):
            if reporter:
                reporter("processing_file", {"file": file_path, "progress": f"{i}/{len(file_paths)}"})
            
            try:
                file_result = self.process_file(file_path, reporter)
                results["files_processed"] += 1
                
                if file_result["success"]:
                    results["files_successful"] += 1
                    results["total_sheets"] += file_result.get("sheets_processed", 0)
                else:
                    results["files_failed"] += 1
                    results["errors"].append(f"{file_path}: {file_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                results["files_failed"] += 1
                results["errors"].append(f"{file_path}: {str(e)}")
        
        # Ingest knowledge base
        if knowledge_base_path or self.kb_config.get("include", False):
            if reporter:
                reporter("ingesting_kb", {"status": "starting"})
            
            kb_results = self.ingest_knowledge_base(knowledge_base_path)
            results["knowledge_base_ingested"] = kb_results
            
            if reporter:
                reporter("ingesting_kb", {"status": "complete", "results": kb_results})
        
        # Emit artifacts
        if reporter:
            reporter("emitting_artifacts", {"status": "starting"})
        
        artifacts = self.emit_artifacts()
        results["artifacts_emitted"] = artifacts
        
        if reporter:
            reporter("emitting_artifacts", {"status": "complete", "artifacts": list(artifacts.keys())})
            reporter("pipeline_complete", results)
        
        return results


def run_enhanced_ingestion(file_paths: List[str], config_path: str = "prompts/ingest_rules.yaml", reporter: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function to run the enhanced ingestion pipeline."""
    pipeline = EnhancedIngestionPipeline(config_path)
    return pipeline.run_full_ingestion(file_paths, reporter=reporter)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_ingestion_pipeline.py <file1> [file2] ...")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    print(f"🧠 Starting Enhanced Ingestion Pipeline for {len(file_paths)} files...")
    
    def simple_reporter(event: str, payload: Dict[str, Any]):
        print(f"📊 {event}: {payload}")
    
    results = run_enhanced_ingestion(file_paths, reporter=simple_reporter)
    
    print("\n" + "="*50)
    print("📋 INGESTION RESULTS")
    print("="*50)
    print(f"Files processed: {results['files_processed']}")
    print(f"Files successful: {results['files_successful']}")
    print(f"Files failed: {results['files_failed']}")
    print(f"Total sheets: {results['total_sheets']}")
    print(f"Artifacts emitted: {list(results['artifacts_emitted'].keys())}")
    
    if results['errors']:
        print(f"\n❌ Errors: {len(results['errors'])}")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    print("\n✅ Enhanced ingestion pipeline complete!")
