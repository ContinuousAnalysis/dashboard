#!/usr/bin/env python3
"""
Generate a PDF with violation plots for each project using LaTeX.
Reads from violations_history.csv and creates plots similar to the dashboard.
"""
import pathlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "dist"
CUMULATIVE_DIR = ROOT / "cumulative_results"

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
    
    # Helper function to load cumulative results for a project
    def load_cumulative_times(project_name):
        """Load end-to-end times from cumulative results CSV for a project."""
        # Try to find matching cumulative CSV file
        # Project name might be "owner/repo" or just "repo"
        repo_name = project_name.split('/')[-1] if '/' in project_name else project_name
        cumulative_csv = CUMULATIVE_DIR / f"{repo_name}-cumulative-results.csv"
        
        if not cumulative_csv.exists():
            return {}
        
        try:
            cum_df = pd.read_csv(cumulative_csv, dtype=str).fillna("")
            cum_df.columns = [c.strip().lower() for c in cum_df.columns]
            
            # Group by commit_sha and algorithm to get times
            times_by_commit = {}
            for _, row in cum_df.iterrows():
                commit_sha = str(row.get('commit_sha', '')).strip()
                algorithm = str(row.get('algorithm', '')).strip().lower()
                e2e_time = parse_time(row.get('end_to_end_time', ''))
                
                if not commit_sha:
                    continue
                
                if commit_sha not in times_by_commit:
                    times_by_commit[commit_sha] = {'original': 0.0, 'pymop': 0.0, 'dylin': 0.0}
                
                if algorithm in ['original', 'pymop', 'dylin']:
                    times_by_commit[commit_sha][algorithm] = e2e_time
            
            return times_by_commit
        except Exception as e:
            print(f"Warning: Could not load cumulative results for {project_name}: {e}")
            return {}
    
    # Generate plot for each project
    project_plots = []
    for project_name, project_df in projects:
        # Preserve the CSV order (top to bottom) - commits appear in CSV in the order they should appear left to right
        # Sort by csv_row_order to ensure we maintain the exact order as they appear in the CSV
        project_df = project_df.sort_values('csv_row_order').reset_index(drop=True)
        
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
            ax.set_ylabel('End-to-End Time (s)', fontsize=12)
        else:
            ax.set_ylabel('Violations', fontsize=12)
        ax.set_title(f'{project_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set x-axis ticks at intervals of 50 (1, 50, 100, 150, ...)
        n = len(x_values)
        if n > 0:
            # Generate tick positions: 0, 49, 99, 149, ... (0-indexed positions)
            # But also include the last commit if it's not already at an interval of 50
            tick_interval = 50
            tick_positions = []
            tick_labels = []
            
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
        
        # Load cumulative times for this project
        cumulative_times = load_cumulative_times(project_name)
        
        # Collect end-to-end times for each commit
        original_times = []
        pymop_times = []
        dylin_times = []
        
        for _, row in project_df.iterrows():
            commit_sha = str(row.get('commit_sha', '')).strip()
            if commit_sha and commit_sha in cumulative_times:
                times = cumulative_times[commit_sha]
                original_times.append(times['original'])
                pymop_times.append(times['pymop'])
                dylin_times.append(times['dylin'])
            else:
                # If no data, use -5 (same as "x")
                original_times.append(-5.0)
                pymop_times.append(-5.0)
                dylin_times.append(-5.0)
        
        # Plot times if version is 'both' or 'times_only'
        if version in ['both', 'times_only']:
            # For times_only, use primary axis, otherwise secondary
            if version == 'times_only':
                ax2 = ax
                # Clear violations plot and redo formatting
                ax.clear()
                ax.set_xlabel('Commit Number', fontsize=12)
                ax.set_ylabel('End-to-End Time (s)', fontsize=12)
                ax.set_title(f'{project_name}', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                # Re-apply x-axis ticks
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
            else:
                ax2 = ax.twinx()
            
            # Plot times
            if len(original_times) > 0:
                ax2.plot(x_values, original_times, marker='s', markersize=2, linewidth=1, 
                        color='#dc2626', linestyle='--', label='Original', alpha=0.7)
                ax2.plot(x_values, pymop_times, marker='^', markersize=2, linewidth=1, 
                        color='#16a34a', linestyle='--', label='Pymop', alpha=0.7)
                ax2.plot(x_values, dylin_times, marker='D', markersize=2, linewidth=1, 
                        color='#fbbf24', linestyle='--', label='Dylin', alpha=0.7)
                
                # Add legend for time plots (1 row x 3 columns)
                # Position legend to avoid blocking data points (lower right)
                ax2.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=3)
                
                # Add note about -5 indicating tool failure (top left, same font size as legend)
                # Position to avoid blocking data points
                ax2.text(0.02, 0.96, 'Note: -5 indicates tool failure', 
                        transform=ax2.transAxes, fontsize=9, 
                        style='italic', color='#6b7280',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#e5e7eb'))
            
            # Set y-axis label
            if version == 'times_only':
                # Already set above when clearing
                ax2.tick_params(axis='y', labelsize=11)
            else:
                ax2.set_ylabel('End-to-End Time (s)', fontsize=12, color='#6b7280')
                ax2.tick_params(axis='y', labelsize=11, labelcolor='#6b7280')
            
            # Format y-axis to show reasonable precision
            from matplotlib.ticker import FuncFormatter
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
            
            # Set y-axis range for times (both times_only and both versions)
            if len(original_times) > 0:
                all_times = original_times + pymop_times + dylin_times
                valid_times = [t for t in all_times if t > -5]
                if valid_times:
                    max_time = max(valid_times)
                    min_time = min(valid_times)
                    # Set maximum to 1.16 * max_time
                    y_max = max_time * 1.16
                    y_min = min(-1, min_time - 1)
                    ax2.set_ylim(y_min, y_max)
                else:
                    # If no valid times, set default range
                    ax2.set_ylim(-1.5, 1.5)
        
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
        
        # For times_only and both versions, calculate average times
        if version in ['times_only', 'both']:
            # Calculate averages (excluding -5 values which indicate failures)
            valid_original = [t for t in original_times if t > -5]
            valid_pymop = [t for t in pymop_times if t > -5]
            valid_dylin = [t for t in dylin_times if t > -5]
            
            avg_original = sum(valid_original) / len(valid_original) if valid_original else 0.0
            avg_pymop = sum(valid_pymop) / len(valid_pymop) if valid_pymop else 0.0
            avg_dylin = sum(valid_dylin) / len(valid_dylin) if valid_dylin else 0.0
            
            # Store averages as a tuple for caption
            time_averages = (avg_original, avg_pymop, avg_dylin)
            project_plots.append((project_name, plot_filename, len(project_df), max_violations, project_slug, time_averages))
        else:
            project_plots.append((project_name, plot_filename, len(project_df), max_violations, project_slug, None))
    
    # Generate LaTeX document
    latex_content = generate_latex_document(project_plots, version)
    
    # Determine output filename based on version
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
        title = "End-to-End Time Plots for Historical Analysis\\\\{\\large 70 GitHub Open Source Projects with Frequent Python File Changes}"
        header_text = "End-to-End Time Plots for Historical Analysis"
    else:  # both
        title = "Violations and End-to-End Time Plots for Historical Analysis\\\\{\\large 70 GitHub Open Source Projects with Frequent Python File Changes}"
        header_text = "Violations and Time Plots for Historical Analysis"
    
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
        if len(plot_data) == 6:
            project_name, _, _, _, project_slug, _ = plot_data
        else:
            project_name, _, _, _, project_slug = plot_data
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
            # Handle both old format (5 items) and new format (6 items with time_averages)
            if len(plot_data) == 6:
                project_name, plot_filename, num_commits, max_violations, project_slug, time_averages = plot_data
            else:
                project_name, plot_filename, num_commits, max_violations, project_slug = plot_data
                time_averages = None
            
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
                    # For times_only, show only average times (no max violations)
                    caption = f"\\footnotesize Commits\\_Count: {num_commits}, \\href{{{dashboard_url}}}{{Link}} \\\\ Avg\\_Time: Orig={avg_orig:.2f}s, Pymop={avg_pymop:.2f}s, Dylin={avg_dylin:.2f}s"
                else:  # both
                    # For both, show max violations and average times
                    caption = f"\\footnotesize Commits\\_Count: {num_commits}, Max\\_Violations: {max_violations}, \\href{{{dashboard_url}}}{{Link}} \\\\ Avg\\_Time: Orig={avg_orig:.2f}s, Pymop={avg_pymop:.2f}s, Dylin={avg_dylin:.2f}s"
            else:
                # For violations_only, show max violations
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
