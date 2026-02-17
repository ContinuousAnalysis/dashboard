#!/usr/bin/env python3
"""
Generate a PDF with violation plots for each project using LaTeX.
Reads from violations_history.csv and creates plots similar to the dashboard.
"""
import pathlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "dist"
CUMULATIVE_DIR = ROOT / "cumulative_results"

ONLY_PASSING = False


def generate_plots_pdf(version='both'):
    """
    Generate a PDF with violation plots for each project.
    
    Args:
        version: 'violations_only', 'both', or 'times_only'
    """
    # Read the violations history CSV
    csv_path = OUT / "violations_history.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Please run export_violations_csv.py first.")
        return
    
    # Read CSV and preserve original row order (top to bottom in CSV)
    df = pd.read_csv(csv_path)
    
    # Add a column to preserve the original CSV row order (top to bottom = 0, 1, 2, ...)
    df['csv_row_order'] = range(len(df))
    
    # Convert timestamp to datetime for reference (but don't use for sorting)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    # Group by project
    projects = df.groupby('project', sort=False)  # sort=False preserves the order projects appear in CSV
    
    # Create temporary directory for plots
    plots_dir = OUT / "plots_temp"
    plots_dir.mkdir(exist_ok=True)
    
    # Helper function to parse time value (treat "x" as -5)
    def parse_time(val):
        if pd.isna(val) or str(val).strip().lower() in ['x', '', 'nan']:
            return -5.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return -5.0
    
    def parse_test_results(val):
        """Parse passed count; return -1 for missing/failure."""
        if pd.isna(val) or str(val).strip().lower() in ['x', '', 'nan']:
            return -1
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return -1

    # Helper function to load cumulative results for a project
    def load_cumulative_data(project_name):
        """Load end-to-end times and passed counts from cumulative results CSV for a project."""
        # Try to find matching cumulative CSV file
        # Project name might be "owner/repo" or just "repo"
        repo_name = project_name.split('/')[-1] if '/' in project_name else project_name
        cumulative_csv = CUMULATIVE_DIR / f"{repo_name}-cumulative-results.csv"
        
        if not cumulative_csv.exists():
            return {}
        
        try:
            cum_df = pd.read_csv(cumulative_csv, dtype=str).fillna("")
            cum_df.columns = [c.strip().lower() for c in cum_df.columns]
            
            # Group by commit_sha and algorithm to get times, passed, failed, errors
            data_by_commit = {}
            for _, row in cum_df.iterrows():
                commit_sha = str(row.get('commit_sha', '')).strip()
                algorithm = str(row.get('algorithm', '')).strip().lower()
                e2e_time = parse_time(row.get('end_to_end_time', ''))
                passed = parse_test_results(row.get('passed', ''))
                failed = parse_test_results(row.get('failed', ''))
                errors = parse_test_results(row.get('errors', ''))
                
                if not commit_sha:
                    continue
                
                if commit_sha not in data_by_commit:
                    data_by_commit[commit_sha] = {
                        'original': 0.0,
                        'pymop': 0.0,
                        'dylin': 0.0,
                        'passed': -1,
                        'failed': -1,
                        'errors': -1,
                    }
                
                if algorithm in ['original', 'pymop', 'dylin']:
                    data_by_commit[commit_sha][algorithm] = e2e_time
                if algorithm == 'original':
                    data_by_commit[commit_sha]['passed'] = passed
                    data_by_commit[commit_sha]['failed'] = failed
                    data_by_commit[commit_sha]['errors'] = errors
            
            return data_by_commit
        except Exception as e:
            print(f"Warning: Could not load cumulative results for {project_name}: {e}")
            return {}
    
    # Generate plot for each project
    project_plots = []
    for project_name, project_df in projects:
        # Preserve the CSV order (top to bottom) - commits appear in CSV in the order they should appear left to right
        # Sort by csv_row_order to ensure we maintain the exact order as they appear in the CSV
        project_df = project_df.sort_values('csv_row_order').reset_index(drop=True)
        
        # Load cumulative data early so we can filter by passed
        cumulative_data = load_cumulative_data(project_name)
        
        # Ignore commits where original's passed is "x" (invalid/missing)
        # and where failed or errors columns are not 0 (when available)
        def keep_commit(sha):
            if not sha:
                return False
            if sha not in cumulative_data:
                return True
            entry = cumulative_data[sha]
            if entry.get('passed', -1) < 0:
                return False
            if ONLY_PASSING and (entry.get('failed', -1) != 0 or entry.get('errors', -1) != 0):
                return False
            return True
        
        mask = project_df['commit_sha'].apply(
            lambda s: keep_commit(str(s).strip()) if pd.notna(s) else False
        )
        project_df = project_df[mask].reset_index(drop=True)

        # Skip projects with no remaining commits after filtering
        if project_df.empty:
            continue
        
        x_values = range(len(project_df))
        y_values = project_df['num_unique_violations'].fillna(0).astype(int).tolist()
        
        # Create plot (adjust height based on version)
        if version in ['times_only', 'both']:
            fig, ax = plt.subplots(figsize=(8, 3.85))  # Slightly reduced height for times/both
        else:
            fig, ax = plt.subplots(figsize=(8, 4))  # Original height for violations_only
        
        # Plot violations if version is 'violations_only' or 'both'
        if version in ['violations_only', 'both']:
            ax.plot(x_values, y_values, marker='o', markersize=2, linewidth=1, color='#2563eb')
        
        # Formatting (larger fonts for better readability)
        ax.set_xlabel('Commit Number', fontsize=12)
        if version == 'times_only':
            ax.set_ylabel('Relative Overhead', fontsize=12)
        else:
            ax.set_ylabel('Violations', fontsize=12)
        ax.set_title(f'{project_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set x-axis ticks at intervals of 50 (1, 50, 100, 150, ...)
        n = len(x_values)
        tick_positions: list[int] = []
        tick_labels: list[str] = []
        if n > 0:
            # Generate tick positions: 0, 49, 99, 149, ... (0-indexed positions)
            # But also include the last commit if it's not already at an interval of 50
            tick_interval = 50
            
            # Start with position 0 (label 1)
            tick_positions.append(0)
            tick_labels.append('1')
            
            # Add ticks at intervals of 50
            pos = tick_interval - 1  # Position 49 (0-indexed) = label 50 (1-indexed)
            while pos < n:
                tick_positions.append(pos)
                tick_labels.append(str(pos + 1))  # Convert to 1-indexed label
                pos += tick_interval
            
            # Add the last commit if it's not already included
            if n > 1 and (n - 1) not in tick_positions:
                tick_positions.append(n - 1)
                tick_labels.append(str(n))
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
        
        # Set y-axis formatting based on version
        from matplotlib.ticker import MaxNLocator, FuncFormatter
        if version == 'times_only':
            # For times, use regular formatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
        else:
            # For violations, use integer ticks
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
        
        # Calculate max violations for the tuple (only if plotting violations)
        if version in ['violations_only', 'both']:
            max_violations = max(y_values) if y_values else 0
            min_violations = min(y_values) if y_values else 0
            
            # Fix y-axis range for flat plots (when all values are the same)
            if max_violations == min_violations:
                padding = 1  # Small padding for flat plots
                ax.set_ylim(min_violations - padding, max_violations + padding)
            else:
                # For both version, use 1.175 * max_violations
                if version == 'both':
                    y_max_violations = max_violations * 1.16
                    padding = 0.5  # Small padding for minimum
                    ax.set_ylim(min_violations - padding, y_max_violations)
                else:
                    # For violations_only, use minimal padding
                    padding = 0.5  # Small padding (0.5 units)
                    ax.set_ylim(min_violations - padding, max_violations + padding)
        else:
            max_violations = 0  # Not used for times_only
        
        # Adjust y-axis label font size
        ax.tick_params(axis='y', labelsize=11)
        
        # Collect times, convert to time/original ratio (original=1 baseline, others=time/original)
        FAILURE_RATIO = -5  # Sentinel for tool failure
        
        def to_ratio(alg_time, orig_time):
            """Compute time/original ratio. Returns FAILURE_RATIO on failure."""
            if orig_time <= -5 or alg_time <= -5 or orig_time <= 0:
                return FAILURE_RATIO
            return alg_time / orig_time
        
        original_ratios = []  # Always 1 (baseline)
        pymop_ratios = []
        dylin_ratios = []
        original_times = []  # Raw times for caption averages
        pymop_times = []
        dylin_times = []
        passed_counts = []  # Number of passing tests (from original)
        
        for _, row in project_df.iterrows():
            commit_sha = str(row.get('commit_sha', '')).strip()
            if commit_sha and commit_sha in cumulative_data:
                times = cumulative_data[commit_sha]
                orig, pymop, dylin = times['original'], times['pymop'], times['dylin']
                original_ratios.append(1.0)  # Original is baseline
                pymop_ratios.append(to_ratio(pymop, orig))
                dylin_ratios.append(to_ratio(dylin, orig))
                original_times.append(orig)
                pymop_times.append(pymop)
                dylin_times.append(dylin)
                passed_counts.append(times.get('passed', -1))
            else:
                original_ratios.append(1.0)
                pymop_ratios.append(FAILURE_RATIO)
                dylin_ratios.append(FAILURE_RATIO)
                original_times.append(-5.0)
                pymop_times.append(-5.0)
                dylin_times.append(-5.0)
                passed_counts.append(-1)
        
        # Plot time ratios if version is 'both' or 'times_only'
        if version in ['both', 'times_only']:
            # For times_only, use primary axis, otherwise secondary
            if version == 'times_only':
                ax2 = ax
                # Clear violations plot and redo formatting
                ax.clear()
                ax.set_xlabel('Commit Number', fontsize=12)
                ax.set_ylabel('Relative Overhead', fontsize=12)
                ax.set_title(f'{project_name}', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                # Re-apply x-axis ticks
                if tick_positions:
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
            else:
                ax2 = ax.twinx()
            
            # Plot time ratios (original=1, others=time/original); use log scale
            if len(original_ratios) > 0:
                # For log scale, replace failure sentinel (-5) with 0.5
                plot_pymop = [r if r > FAILURE_RATIO else 0.5 for r in pymop_ratios]
                plot_dylin = [r if r > FAILURE_RATIO else 0.5 for r in dylin_ratios]
                ax2.set_yscale('log')
                ax2.plot(x_values, original_ratios, marker='s', markersize=2, linewidth=1, 
                        color='#dc2626', linestyle='--', label='Original (1)', alpha=0.7)
                ax2.plot(x_values, plot_pymop, marker='^', markersize=2, linewidth=1, 
                        color='#16a34a', linestyle='--', label='Pymop', alpha=0.7)
                ax2.plot(x_values, plot_dylin, marker='D', markersize=2, linewidth=1, 
                        color='#fbbf24', linestyle='--', label='Dylin', alpha=0.7)
                
                # Add legend for time plots (1 row x 3 columns)
                ax2.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=3)
                
                # Add note about failure indicator
                ax2.text(0.02, 0.96, 'Note: 0.5 indicates tool failure', 
                        transform=ax2.transAxes, fontsize=9, 
                        style='italic', color='#6b7280',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#e5e7eb'))
            
            # Set y-axis label
            if version == 'times_only':
                ax2.tick_params(axis='y', labelsize=11)
            else:
                ax2.set_ylabel('Relative Overhead', fontsize=12, color='#6b7280')
                ax2.tick_params(axis='y', labelsize=11, labelcolor='#6b7280')
            
            # Format y-axis
            from matplotlib.ticker import FuncFormatter
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
            
            # Set y-axis range for log scale (positive values only)
            if len(original_ratios) > 0:
                valid_ratios = [r for r in (original_ratios + pymop_ratios + dylin_ratios) if r > FAILURE_RATIO]
                if valid_ratios:
                    max_r = max(valid_ratios)
                    min_r = min(valid_ratios)
                    y_max = max_r * 2
                    y_min = max(0.1, min_r / 2)
                    ax2.set_ylim(y_min, y_max)
                else:
                    ax2.set_ylim(0.1, 2)
        
        # Add passing tests line (secondary y-axis on the left, offset)
        valid_passed = [p for p in passed_counts if p >= 0]
        if valid_passed:
            # Use NaN for missing so line breaks at gaps
            plot_passed = [p if p >= 0 else float('nan') for p in passed_counts]
            ax_passed = ax.twinx()
            ax_passed.spines['right'].set_visible(False)
            ax_passed.spines['left'].set_position(('outward', 55))
            ax_passed.yaxis.set_ticks_position('left')
            ax_passed.yaxis.set_label_position('left')
            ax_passed.plot(x_values, plot_passed, marker='.', markersize=1.5, linewidth=0.8,
                          color='#9333ea', linestyle=':', label='Passed', alpha=0.8)
            ax_passed.set_ylabel('Tests Passed', fontsize=11, color='#9333ea')
            ax_passed.tick_params(axis='y', labelsize=9, labelcolor='#9333ea')
            ax_passed.yaxis.set_major_locator(MaxNLocator(integer=True))
            y_max_p = max(valid_passed) * 1.25 if max(valid_passed) > 2 else 2.5
            y_min_p = -2.5
            ax_passed.set_ylim(y_min_p, y_max_p)
        
        plt.tight_layout()
        
        # Generate project slug for dashboard URL
        project_slug = project_name.replace("/", "-").lower()
        
        # Save plot with version suffix
        base_name = project_name.replace('/', '-').replace(' ', '_')
        if version == 'violations_only':
            plot_filename = f"{base_name}_violations.png"
        elif version == 'times_only':
            plot_filename = f"{base_name}_times.png"
        else:
            plot_filename = f"{base_name}_both.png"
        plot_path = plots_dir / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # For times_only and both versions, calculate average times (for caption)
        if version in ['times_only', 'both']:
            # Calculate average time in seconds (excluding failure sentinel -5)
            valid_original = [t for t in original_times if t > -5]
            valid_pymop = [t for t in pymop_times if t > -5]
            valid_dylin = [t for t in dylin_times if t > -5]
            
            avg_original = sum(valid_original) / len(valid_original) if valid_original else 0.0
            avg_pymop = sum(valid_pymop) / len(valid_pymop) if valid_pymop else 0.0
            avg_dylin = sum(valid_dylin) / len(valid_dylin) if valid_dylin else 0.0
            
            # Store average times (s) and avg passed for caption
            valid_p = [p for p in passed_counts if p >= 0]
            avg_passed = sum(valid_p) / len(valid_p) if valid_p else 0
            time_averages = (avg_original, avg_pymop, avg_dylin)
            project_plots.append((project_name, plot_filename, len(project_df), max_violations, project_slug, time_averages, avg_passed))
        else:
            valid_p = [p for p in passed_counts if p >= 0]
            avg_passed = sum(valid_p) / len(valid_p) if valid_p else 0
            project_plots.append((project_name, plot_filename, len(project_df), max_violations, project_slug, None, avg_passed))
    
    # Generate LaTeX document
    latex_content = generate_latex_document(project_plots, version)
    
    # Determine output filename based on version
    if ONLY_PASSING:
        if version == 'violations_only':
            base_name = "violations_plots_violations_only_only_passing"
        elif version == 'times_only':
            base_name = "violations_plots_times_only_only_passing"
        else:
            base_name = "violations_plots_both_only_passing"
    else:
        if version == 'violations_only':
            base_name = "violations_plots_violations_only"
        elif version == 'times_only':
            base_name = "violations_plots_times_only"
        else:
            base_name = "violations_plots_both"
    
    # Write LaTeX file
    tex_path = OUT / f"{base_name}.tex"
    tex_path.write_text(latex_content, encoding='utf-8')
    
    # Compile to PDF
    print(f"Generated LaTeX file: {tex_path}")
    print("Compiling to PDF...")
    
    try:
        # Run pdflatex twice for proper references
        for i in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(OUT), str(tex_path)],
                capture_output=True,
                text=True,
                cwd=str(OUT)
            )
            if result.returncode != 0:
                print(f"Warning: pdflatex run {i+1} had issues:")
                print(result.stderr)
        
        pdf_path = OUT / f"{base_name}.pdf"
        if pdf_path.exists():
            print(f"✓ Successfully generated PDF: {pdf_path}")
            
            # Clean up auxiliary files
            for ext in ['.aux', '.log', '.out']:
                aux_file = OUT / f"{base_name}{ext}"
                if aux_file.exists():
                    aux_file.unlink()
        else:
            print("ERROR: PDF was not generated. Check LaTeX errors above.")
            
    except FileNotFoundError:
        print("ERROR: pdflatex not found. Please install LaTeX (e.g., texlive or miktex)")
        print(f"LaTeX source file is available at: {tex_path}")
        print("You can compile it manually with: pdflatex violations_plots.tex")
    
    # Optionally clean up plots directory (comment out if you want to keep them)
    # shutil.rmtree(plots_dir)

def generate_latex_document(project_plots, version='both'):
    """
    Generate LaTeX document content with all project plots.
    
    Args:
        project_plots: List of project plot data
        version: 'violations_only', 'both', or 'times_only'
    """
    # Sort projects alphabetically
    project_plots = sorted(project_plots, key=lambda x: x[0])

    plots_per_page = 10
    
    latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{subcaption}
\usepackage{caption}
\usepackage{multicol}
\usepackage{titling}

% Pull entire title block up
\setlength{\droptitle}{-1.5cm}

% Reduce space between title and date
\posttitle{\par\vspace{-1cm}\end{center}}
\postdate{\par\vspace{-0.5cm}\end{center}}

\captionsetup[subfigure]{labelformat=empty}

\geometry{margin=0.75in}
\pagestyle{fancy}
"""
    
    # Set title and header based on version
    if version == 'violations_only':
        title = "Violations Plots for Historical Analysis\\\\{\\large 70 GitHub Open Source Projects with Frequent Python File Changes}"
        header_text = "Violations Plots for Historical Analysis"
    elif version == 'times_only':
        title = "Relative Overhead Plots for Historical Analysis\\\\{\\large 70 GitHub Open Source Projects with Frequent Python File Changes}"
        header_text = "Relative Overhead Plots for Historical Analysis"
    else:  # both
        title = "Violations and Relative Overhead Plots for Historical Analysis\\\\{\\large 70 GitHub Open Source Projects with Frequent Python File Changes}"
        header_text = "Violations and Relative Overhead Plots for Historical Analysis"
    
    latex += f"""\\fancyhf{{}}
\\fancyhead[L]{{{header_text}}}
\\fancyhead[R]{{\\today}}
\\fancyfoot[C]{{\\thepage}}

\\title{{{title}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\hrule
\\vspace{{0.65cm}}

{{\\normalsize\\textbf{{Project List}}}}
\\vspace{{-0.2cm}}
\\begin{{multicols}}{{2}}
\\small
\\setlength{{\\itemsep}}{{0.1em}}
\\begin{{enumerate}}
"""
    
    # Create project list with links
    for plot_data in project_plots:
        project_name = plot_data[0]
        project_slug = plot_data[4]
        safe_name = (project_name
                    .replace('\\', r'\textbackslash{}')
                    .replace('_', r'\_')
                    .replace('&', r'\&')
                    .replace('%', r'\%')
                    .replace('#', r'\#'))
        latex += f"\\item \\hyperlink{{proj:{project_slug}}}{{{safe_name}}}\n"
    
    latex += r"""\end{enumerate}
\normalsize
\end{multicols}
\newpage

"""
    
    # Group plots 10 per page (2 columns x 5 rows)
    for page_idx in range(0, len(project_plots), plots_per_page):
        page_plots = project_plots[page_idx:page_idx + plots_per_page]
        
        # Calculate page number (first plot page is page 2, since title is page 1)
        page_num = page_idx // plots_per_page + 2
        
        # Add newpage only if not the first plot page (first page already has newpage from project list)
        if page_idx > 0:
            latex += r"\newpage"
        
        latex += f"""
\\phantomsection
\\label{{page{page_num}}}
\\begin{{figure}}[h]
\\centering
"""
        
        # Create 2x5 grid using subfigure
        for i, plot_data in enumerate(page_plots):
            # Handle 5, 6, or 7 items (avg_passed added)
            if len(plot_data) >= 7:
                project_name, plot_filename, num_commits, max_violations, project_slug, time_averages, avg_passed = plot_data[:7]
            elif len(plot_data) == 6:
                project_name, plot_filename, num_commits, max_violations, project_slug, time_averages = plot_data
                avg_passed = 0
            else:
                project_name, plot_filename, num_commits, max_violations, project_slug = plot_data
                time_averages = None
                avg_passed = 0
            
            # Determine position: 2 columns, 5 rows
            col = i % 2
            row = i // 2
            
            if col == 0:
                latex += r"\begin{subfigure}[b]{0.49\textwidth}"
            else:
                latex += r"\hfill\begin{subfigure}[b]{0.49\textwidth}"

            dashboard_url = f"https://continuousanalysis.github.io/dashboard/\\#/hist/p/{project_slug}"
            
            # Build caption based on version
            if time_averages is not None:
                avg_orig, avg_pymop, avg_dylin = time_averages
                if version == 'times_only':
                    caption = f"\\footnotesize Commits\\_Count: {num_commits}, \\href{{{dashboard_url}}}{{Link}} \\\\ Avg\\_Time: Orig={avg_orig:.2f}s, Pymop={avg_pymop:.2f}s, Dylin={avg_dylin:.2f}s"
                else:  # both
                    caption = f"\\footnotesize Commits\\_Count: {num_commits}, Max\\_Violations: {max_violations}, \\href{{{dashboard_url}}}{{Link}} \\\\ Avg\\_Time: Orig={avg_orig:.2f}s, Pymop={avg_pymop:.2f}s, Dylin={avg_dylin:.2f}s"
            else:
                caption = f"\\footnotesize Commits\\_Count: {num_commits}, Max\\_Violations: {max_violations}, \\href{{{dashboard_url}}}{{Link}}"

            latex += f"""
\\centering
\\hypertarget{{proj:{project_slug}}}{{}}%
\\includegraphics[width=\\textwidth]{{plots_temp/{plot_filename}}}
\\caption{{{caption}}}
\\end{{subfigure}}
"""
            
            # Add line break after every 2 plots (end of row)
            if col == 1:
                latex += r"\\[0.2cm]"
        
        latex += r"""
\end{figure}
\newpage

"""
    
    latex += r"""
\end{document}
"""
    
    return latex

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "only-passing":
            ONLY_PASSING = True
        else:
            raise ValueError(f"Invalid argument: {sys.argv[1]}. Expected 'only-passing' or nothing.")

    # Generate three PDFs: violations only, both, and times only
    print("=" * 60)
    print("Generating PDFs for all versions...")
    print("=" * 60)
    
    print("\n1. Generating violations-only PDF...")
    generate_plots_pdf(version='violations_only')
    
    print("\n2. Generating both violations and times PDF...")
    generate_plots_pdf(version='both')
    
    print("\n3. Generating times-only PDF...")
    generate_plots_pdf(version='times_only')
    
    print("\n" + "=" * 60)
    print("All PDFs generated successfully!")
    print("=" * 60)
