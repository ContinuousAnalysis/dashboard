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

def generate_plots_pdf():
    """
    Generate a PDF with violation plots for each project.
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
    
    # Generate plot for each project
    project_plots = []
    for project_name, project_df in projects:
        # Preserve the CSV order (top to bottom) - commits appear in CSV in the order they should appear left to right
        # Sort by csv_row_order to ensure we maintain the exact order as they appear in the CSV
        project_df = project_df.sort_values('csv_row_order').reset_index(drop=True)
        
        x_values = range(len(project_df))
        x_labels = [sha[:7] if pd.notna(sha) else str(i) for i, sha in enumerate(project_df['commit_sha'])]
        
        y_values = project_df['num_unique_violations'].fillna(0).astype(int).tolist()
        
        # Create plot (larger size for better visibility)
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Plot line with points
        ax.plot(x_values, y_values, marker='o', markersize=2, linewidth=1, color='#2563eb')
        
        # Formatting (larger fonts for better readability)
        ax.set_xlabel('Commit', fontsize=12)
        ax.set_ylabel('Violations', fontsize=12)
        ax.set_title(f'{project_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set x-axis ticks (show first, quarter, half, three-quarter, last)
        n = len(x_values)
        if n > 1:
            tick_indices = [0, n//4, n//2, 3*n//4, n-1]
            tick_indices = sorted(set(tick_indices))  # Remove duplicates
            ax.set_xticks([x_values[i] for i in tick_indices])
            ax.set_xticklabels([x_labels[i] for i in tick_indices], rotation=0, fontsize=10)
        elif n == 1:
            ax.set_xticks([0])
            ax.set_xticklabels([x_labels[0]], fontsize=10)
        
        # Set y-axis to show only integer ticks
        from matplotlib.ticker import MaxNLocator, FuncFormatter
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Format y-axis labels as integers (no decimals)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
        
        # Calculate max violations for the tuple
        max_violations = max(y_values) if y_values else 0
        
        # Adjust y-axis label font size
        ax.tick_params(axis='y', labelsize=11)
        
        plt.tight_layout()
        
        # Generate project slug for dashboard URL
        project_slug = project_name.replace("/", "-").lower()
        
        # Save plot
        plot_filename = f"{project_name.replace('/', '-').replace(' ', '_')}.png"
        plot_path = plots_dir / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        project_plots.append((project_name, plot_filename, len(project_df), max_violations, project_slug))
    
    # Generate LaTeX document
    latex_content = generate_latex_document(project_plots)
    
    # Write LaTeX file
    tex_path = OUT / "violations_plots.tex"
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
        
        pdf_path = OUT / "violations_plots.pdf"
        if pdf_path.exists():
            print(f"✓ Successfully generated PDF: {pdf_path}")
            
            # Clean up auxiliary files
            for ext in ['.aux', '.log', '.out']:
                aux_file = OUT / f"violations_plots{ext}"
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

def generate_latex_document(project_plots):
    """
    Generate LaTeX document content with all project plots.
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
\fancyhf{}
\fancyhead[L]{Violations Plots for Historical Analysis}
\fancyhead[R]{\today}
\fancyfoot[C]{\thepage}

\title{Continuous Dynamic Analysis: Violations Plots for Historical Analysis\\{\large 70 GitHub Open Source Projects with Frequent Python File Changes}}
\date{\today}

\begin{document}

\maketitle

\hrule
\vspace{0.65cm}

{\normalsize\textbf{Project List}}
\vspace{-0.2cm}
\begin{multicols}{2}
\small
\setlength{\itemsep}{0.1em}
\begin{enumerate}
"""
    
    # Create project list with links
    for project_name, _, _, _, project_slug in project_plots:
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
        for i, (project_name, plot_filename, num_commits, max_violations, project_slug) in enumerate(page_plots):
            # Determine position: 2 columns, 5 rows
            col = i % 2
            row = i // 2
            
            if col == 0:
                latex += r"\begin{subfigure}[b]{0.49\textwidth}"
            else:
                latex += r"\hfill\begin{subfigure}[b]{0.49\textwidth}"

            dashboard_url = f"https://continuousanalysis.github.io/dashboard/\\#/hist/p/{project_slug}"

            latex += f"""
\\centering
\\hypertarget{{proj:{project_slug}}}{{}}%
\\includegraphics[width=\\textwidth]{{plots_temp/{plot_filename}}}
\\caption{{\\footnotesize Commits: {num_commits}, Max\\_Violations: {max_violations}, \\href{{{dashboard_url}}}{{Dashboard\\_Link}}}}
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
    generate_plots_pdf()

