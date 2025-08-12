# charting.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np

# Enhanced foundation imports
try:
    from constants import DATA_ROOT
    from path_utils import canon_path
    from logger import log_event
except ImportError:
    # Fallback for standalone usage
    DATA_ROOT = "/Project_Root/04_Data"
    def canon_path(p): return str(Path(p).resolve())
    def log_event(msg, path=None): print(f"LOG: {msg}")

# AI integration for intelligent charting
try:
    from llm_client import get_openai_client, chat_completion
except ImportError:
    get_openai_client = None
    chat_completion = None

# Cloud storage for chart uploads
try:
    from dbx_utils import upload_bytes
except ImportError:
    def upload_bytes(path, data, mode="overwrite"): 
        # Fallback to local save
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

# ================== ENTERPRISE CHARTING WITH AI ENHANCEMENT ==================

def save_bar_chart(data: dict, title: str, x_label: str, y_label: str, output_path: str, 
                  ai_enhance: bool = True) -> str:
    """
    Enhanced bar chart generation with AI-powered insights and enterprise formatting.
    
    Args:
        data: Keys are x-axis labels, values are heights
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label  
        output_path: Full path to save the image
        ai_enhance: Enable AI-powered chart recommendations
        
    Returns:
        Path to saved chart
    """
    if not data:
        log_event("No data provided for bar chart")
        return ""

    # AI-enhanced chart styling and insights
    chart_config = _get_ai_chart_config(data, "bar", title) if ai_enhance else {}
    
    # Enterprise-grade styling
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=chart_config.get('figsize', (12, 8)))
    
    # Create bar chart with enhanced styling
    bars = ax.bar(data.keys(), data.values(), 
                 color=chart_config.get('color', '#2E86AB'),
                 alpha=0.8,
                 edgecolor='white',
                 linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:,.0f}' if height > 1000 else f'{height:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Enhanced formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    
    # Format y-axis for financial data
    if any(term in y_label.lower() for term in ['cost', 'value', 'price', 'amount']):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Rotate x-labels if too many or too long
    if len(data) > 8 or max(len(str(k)) for k in data.keys()) > 10:
        plt.xticks(rotation=45, ha="right")
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save to canonical path
    output_path = _ensure_charts_folder(output_path)
    
    try:
        if output_path.startswith('/') or 'dropbox' in output_path.lower():
            # Cloud save
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            upload_bytes(output_path, buf.getvalue())
        else:
            # Local save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        log_event(f"Bar chart saved: {output_path}")
        
    except Exception as e:
        log_event(f"Failed to save chart: {e}")
    finally:
        plt.close(fig)
    
    return output_path

def save_scatter_plot(x, y, x_label, y_label, title, output_path, 
                     ai_enhance: bool = True, **kwargs) -> str:
    """
    Enhanced scatter plot with AI-powered insights and trend analysis.
    """
    # AI-enhanced configuration
    chart_config = _get_ai_chart_config({"x": x, "y": y}, "scatter", title) if ai_enhance else {}
    
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=chart_config.get('figsize', (10, 8)))
    
    # Create scatter plot with enhanced styling
    scatter = ax.scatter(x, y, 
                        alpha=0.7, 
                        s=kwargs.get('s', 60),
                        c=chart_config.get('color', '#A23B72'),
                        edgecolors='white',
                        linewidth=0.5)
    
    # Add trend line if AI recommends it
    if chart_config.get('add_trendline', True) and len(x) > 3:
        _add_trendline(ax, x, y)
    
    # Enhanced formatting
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Format axes for financial data
    if any(term in x_label.lower() for term in ['cost', 'value', 'price', 'amount']):
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    if any(term in y_label.lower() for term in ['cost', 'value', 'price', 'amount']):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to canonical path
    output_path = _ensure_charts_folder(output_path)
    
    try:
        if output_path.startswith('/') or 'dropbox' in output_path.lower():
            # Cloud save
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            upload_bytes(output_path, buf.getvalue())
        else:
            # Local save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        log_event(f"Scatter plot saved: {output_path}")
        
    except Exception as e:
        log_event(f"Failed to save chart: {e}")
    finally:
        plt.close(fig)
    
    return output_path

def create_supply_chain_dashboard(df: pd.DataFrame, title: str = "Supply Chain Dashboard", 
                                 output_path: Optional[str] = None) -> str:
    """
    Create a comprehensive supply chain dashboard with multiple charts.
    """
    if output_path is None:
        output_path = f"{DATA_ROOT}/02_EDA_Charts/supply_chain_dashboard.png"
    
    output_path = _ensure_charts_folder(output_path)
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    try:
        # Chart 1: Value distribution
        if 'extended_cost' in df.columns or 'total_cost' in df.columns:
            value_col = 'extended_cost' if 'extended_cost' in df.columns else 'total_cost'
            ax1 = axes[0, 0]
            df[value_col].hist(bins=30, ax=ax1, alpha=0.7, color='#2E86AB')
            ax1.set_title('Value Distribution', fontweight='bold')
            ax1.set_xlabel('Value ($)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        
        # Chart 2: Top items by value
        if 'part_no' in df.columns and ('extended_cost' in df.columns or 'total_cost' in df.columns):
            value_col = 'extended_cost' if 'extended_cost' in df.columns else 'total_cost'
            ax2 = axes[0, 1]
            top_parts = df.groupby('part_no')[value_col].sum().sort_values(ascending=False).head(10)
            top_parts.plot(kind='bar', ax=ax2, color='#A23B72')
            ax2.set_title('Top 10 Parts by Value', fontweight='bold')
            ax2.set_xlabel('Part Number')
            ax2.set_ylabel('Total Value ($)')
            ax2.tick_params(axis='x', rotation=45)
        
        # Chart 3: Quantity vs Value scatter
        if all(col in df.columns for col in ['qty_on_hand', 'extended_cost']):
            ax3 = axes[1, 0]
            ax3.scatter(df['qty_on_hand'], df['extended_cost'], alpha=0.6, color='#F18F01')
            ax3.set_title('Quantity vs Value', fontweight='bold')
            ax3.set_xlabel('Quantity on Hand')
            ax3.set_ylabel('Extended Cost ($)')
            ax3.grid(True, alpha=0.3)
        
        # Chart 4: Location analysis
        if 'location' in df.columns and 'extended_cost' in df.columns:
            ax4 = axes[1, 1]
            location_totals = df.groupby('location')['extended_cost'].sum().sort_values(ascending=False)
            location_totals.plot(kind='bar', ax=ax4, color='#C73E1D')
            ax4.set_title('Value by Location', fontweight='bold')
            ax4.set_xlabel('Location')
            ax4.set_ylabel('Total Value ($)')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save dashboard
        if output_path.startswith('/') or 'dropbox' in output_path.lower():
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            upload_bytes(output_path, buf.getvalue())
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        log_event(f"Dashboard saved: {output_path}")
        
    except Exception as e:
        log_event(f"Failed to create dashboard: {e}")
    finally:
        plt.close(fig)
    
    return output_path

# ================== HELPER FUNCTIONS ==================

def _ensure_charts_folder(output_path: str) -> str:
    """Ensure chart is saved to the correct enterprise folder structure."""
    
    # If path doesn't include charts folder, add it
    if '02_EDA_Charts' not in output_path:
        charts_folder = f"{DATA_ROOT}/02_EDA_Charts"
        filename = Path(output_path).name
        output_path = f"{charts_folder}/{filename}"
    
    return canon_path(output_path)

def _get_ai_chart_config(data: Union[Dict, List], chart_type: str, title: str) -> Dict[str, Any]:
    """Get AI-powered chart configuration and styling recommendations."""
    
    if not get_openai_client or not chat_completion:
        return {"color": "#2E86AB", "figsize": (12, 8)}
    
    try:
        client = get_openai_client()
        
        # Prepare data summary for AI
        if isinstance(data, dict):
            data_summary = {
                "type": "dictionary",
                "length": len(data),
                "sample_keys": list(data.keys())[:5],
                "sample_values": list(data.values())[:5],
                "max_value": max(data.values()) if data else 0,
                "min_value": min(data.values()) if data else 0
            }
        else:
            data_summary = {
                "type": "array",
                "length": len(data),
                "sample_data": data[:5] if data else []
            }
        
        prompt = f"""
        You are a data visualization expert. Analyze this {chart_type} chart request and provide styling recommendations.
        
        Chart: {title}
        Data Summary: {data_summary}
        
        Provide recommendations in JSON format:
        {{
            "color": "#hex_color",
            "figsize": [width, height],
            "add_trendline": true/false,
            "style_recommendations": ["recommendation1", "recommendation2"],
            "business_insights": ["insight1", "insight2"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a data visualization expert specializing in supply chain analytics. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = chat_completion(client, messages, model="gpt-4o-mini")
        
        try:
            config = eval(response)  # Parse JSON response
            return config
        except:
            # Fallback configuration
            return {"color": "#2E86AB", "figsize": (12, 8)}
            
    except Exception as e:
        log_event(f"AI chart config failed: {e}")
        return {"color": "#2E86AB", "figsize": (12, 8)}

def _add_trendline(ax, x, y):
    """Add a trend line to a scatter plot."""
    try:
        # Convert to numpy arrays and filter out NaN values
        x_clean = np.array(x)
        y_clean = np.array(y)
        
        # Remove NaN values
        mask = ~(np.isnan(x_clean) | np.isnan(y_clean))
        x_clean = x_clean[mask]
        y_clean = y_clean[mask]
        
        if len(x_clean) > 1:
            # Calculate trend line
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            
            # Plot trend line
            ax.plot(x_clean, p(x_clean), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax.legend()
            
    except Exception as e:
        log_event(f"Failed to add trend line: {e}")
