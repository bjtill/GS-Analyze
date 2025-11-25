#!/usr/bin/env python3
"""
Comprehensive Genotype Data Analysis Program
Analyzes genotype data in AB format with marker keys and generates HTML reports
Version 2.5: Added numbered comparison pairs for custom marker concordance plots and --exclude-failed-markers option
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import sys
from pathlib import Path
from datetime import datetime
import json


class GenotypeAnalyzer:
    """Main class for genotype data analysis"""
    
    def __init__(self, rawdata_file, markerkey_file, report_name, sample_call_rate_threshold, 
                 run_concordance=False, custom_sample_groups=None, custom_marker_groups=None,
                 exclude_failed_markers=False):
        """
        Initialize the analyzer with input parameters
        
        Args:
            rawdata_file: Path to raw genotype data file (marker, sample, genotype)
            markerkey_file: Path to marker key file (marker, A_allele, B_allele)
            report_name: Name for the output report
            sample_call_rate_threshold: Minimum sample call rate (0-100)
            run_concordance: Whether to run sample concordance analysis
            custom_sample_groups: Path to file with custom sample groups for concordance
            custom_marker_groups: Path to file with custom marker groups for concordance
            exclude_failed_markers: Exclude markers with 0% call rate from sample call rate calculations
        """
        self.rawdata_file = Path(rawdata_file)
        self.markerkey_file = Path(markerkey_file)
        self.report_name = report_name
        self.sample_call_rate_threshold = sample_call_rate_threshold / 100.0  # Convert to decimal
        self.run_concordance = run_concordance
        self.custom_sample_groups = Path(custom_sample_groups) if custom_sample_groups else None
        self.custom_marker_groups = Path(custom_marker_groups) if custom_marker_groups else None
        self.exclude_failed_markers = exclude_failed_markers
        
        # Output directory
        self.output_dir = Path(f"{report_name}_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.rawdata = None
        self.markerkey = None
        self.filtered_data = None
        self.sample_call_rates = None
        self.marker_call_rates = None
        self.passed_samples = None
        self.failed_samples = None
        self.sample_b_freq = None
        self.marker_b_freq = None
        self.concordance_passing = None
        self.concordance_with_failed = None
        self.custom_sample_concordance_passing = None
        self.custom_sample_concordance_with_failed = None
        self.custom_marker_concordance_passing = None
        self.custom_marker_concordance_with_failed = None
        
        # Analysis timestamp
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Command line used
        self.command_line = None
    def convert_genotype_to_alleles(self, marker, genotype):
        """
        Convert AB format genotype to actual alleles (G/T format)
        
        Args:
            marker: Marker name
            genotype: Genotype in AB format (AA, AB, BB, NC)
            
        Returns:
            Genotype in G/T format (e.g., A/A, G/T, etc.) or NC
        """
        if genotype == 'NC' or pd.isna(genotype):
            return 'NC'
        
        # Get the marker key
        marker_info = self.markerkey[self.markerkey['Marker'] == marker]
        if len(marker_info) == 0:
            return 'NC'  # Marker not found
        
        a_allele = marker_info.iloc[0]['A_Allele']
        b_allele = marker_info.iloc[0]['B_Allele']
        
        # Convert based on genotype
        if genotype == 'AA':
            return f"{a_allele}/{a_allele}"
        elif genotype == 'AB':
            return f"{a_allele}/{b_allele}"
        elif genotype == 'BB':
            return f"{b_allele}/{b_allele}"
        else:
            return 'NC'
    
    def load_data(self):
        """Load and parse input data files"""
        print("Loading data files...")
        
        # Load raw data - handle tab or space delimited with 3 columns
        print(f"  Loading {self.rawdata_file}...")
        try:
            # Read the file and parse it properly
            with open(self.rawdata_file, 'r') as f:
                lines = f.readlines()
            
            # Parse each line into marker, sample, genotype
            data_rows = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    marker = parts[0]
                    sample = parts[1]
                    genotype = parts[2]
                    data_rows.append([marker, sample, genotype])
            
            self.rawdata = pd.DataFrame(data_rows, columns=['Marker', 'Sample', 'Genotype'])
            print(f"    Loaded {len(self.rawdata)} genotype records")
            
        except Exception as e:
            print(f"Error loading raw data: {e}")
            sys.exit(1)
        
        # Load marker key
        print(f"  Loading {self.markerkey_file}...")
        try:
            with open(self.markerkey_file, 'r') as f:
                lines = f.readlines()
            
            marker_rows = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    marker_rows.append([parts[0], parts[1], parts[2]])
            
            self.markerkey = pd.DataFrame(marker_rows, columns=['Marker', 'A_Allele', 'B_Allele'])
            print(f"    Loaded {len(self.markerkey)} markers")
            
        except Exception as e:
            print(f"Error loading marker key: {e}")
            sys.exit(1)
        
        # Filter rawdata to only include markers in the marker key
        markers_of_interest = set(self.markerkey['Marker'])
        self.filtered_data = self.rawdata[self.rawdata['Marker'].isin(markers_of_interest)].copy()
        print(f"  Filtered to {len(self.filtered_data)} records matching {len(markers_of_interest)} markers")
        
    def calculate_sample_call_rates(self):
        """Calculate call rates for each sample"""
        print("\nCalculating sample call rates...")
        
        # First, identify completely failed markers (0% call rate across all samples) if option is set
        failed_markers_set = set()
        if self.exclude_failed_markers:
            print("  Identifying completely failed markers (0% call rate)...")
            for marker in self.markerkey['Marker']:
                marker_data = self.filtered_data[self.filtered_data['Marker'] == marker]
                successful_calls = len(marker_data[
                    (marker_data['Genotype'] != 'NC') & 
                    (marker_data['Genotype'].notna())
                ])
                if successful_calls == 0:
                    failed_markers_set.add(marker)
            
            if len(failed_markers_set) > 0:
                print(f"  Found {len(failed_markers_set)} completely failed markers (will be excluded from denominator)")
            else:
                print(f"  No completely failed markers found")
        
        # Calculate total markers for denominator
        if self.exclude_failed_markers:
            total_markers = len(self.markerkey) - len(failed_markers_set)
            print(f"  Using {total_markers} markers for call rate calculation ({len(self.markerkey)} total - {len(failed_markers_set)} failed)")
        else:
            total_markers = len(self.markerkey)
            print(f"  Using {total_markers} markers for call rate calculation")
        
        samples = self.filtered_data['Sample'].unique()
        
        results = []
        for sample in samples:
            sample_data = self.filtered_data[self.filtered_data['Sample'] == sample]
            
            # Count successful calls (not NC and not missing)
            successful_calls = len(sample_data[
                (sample_data['Genotype'] != 'NC') & 
                (sample_data['Genotype'].notna())
            ])
            
            # If excluding failed markers, adjust successful calls
            if self.exclude_failed_markers and len(failed_markers_set) > 0:
                # Subtract any successful calls that were for completely failed markers
                # (though this should be 0 by definition)
                successful_markers = set(sample_data[
                    (sample_data['Genotype'] != 'NC') & 
                    (sample_data['Genotype'].notna())
                ]['Marker'])
                # Remove completely failed markers from consideration
                successful_calls = len(successful_markers - failed_markers_set)
            
            # Failed calls
            failed_calls = total_markers - successful_calls
            
            # Call rate
            call_rate = successful_calls / total_markers if total_markers > 0 else 0
            
            # Get list of failed markers
            present_markers = set(sample_data[
                (sample_data['Genotype'] != 'NC') & 
                (sample_data['Genotype'].notna())
            ]['Marker'])
            
            all_markers = set(self.markerkey['Marker'])
            # If excluding failed markers, don't count them in the failed list
            if self.exclude_failed_markers:
                all_markers = all_markers - failed_markers_set
            failed_markers = list(all_markers - present_markers)
            
            results.append({
                'Sample': sample,
                'Total_Markers_With_Call': successful_calls,
                'Total_Failed': failed_calls,
                'Call_Rate': call_rate,
                'Failed_Markers': ', '.join(failed_markers) if failed_markers else 'None'
            })
        
        self.sample_call_rates = pd.DataFrame(results)
        self.sample_call_rates = self.sample_call_rates.sort_values('Call_Rate', ascending=False)
        
        print(f"  Calculated call rates for {len(self.sample_call_rates)} samples")
        print(f"  Call rate range: {self.sample_call_rates['Call_Rate'].min():.2%} - {self.sample_call_rates['Call_Rate'].max():.2%}")
        
    def calculate_marker_call_rates(self):
        """Calculate call rates for each marker"""
        print("\nCalculating marker call rates...")
        
        total_samples = len(self.filtered_data['Sample'].unique())
        markers = self.markerkey['Marker'].unique()
        
        results = []
        for marker in markers:
            marker_data = self.filtered_data[self.filtered_data['Marker'] == marker]
            
            # Count successful calls
            successful_calls = len(marker_data[
                (marker_data['Genotype'] != 'NC') & 
                (marker_data['Genotype'].notna())
            ])
            
            failed_calls = total_samples - successful_calls
            call_rate = successful_calls / total_samples if total_samples > 0 else 0
            
            # Get list of failed samples for this marker
            present_samples = set(marker_data[
                (marker_data['Genotype'] != 'NC') & 
                (marker_data['Genotype'].notna())
            ]['Sample'])
            
            all_samples = set(self.filtered_data['Sample'].unique())
            failed_samples = list(all_samples - present_samples)
            
            results.append({
                'Marker': marker,
                'Total_Samples_With_Call': successful_calls,
                'Total_Failed': failed_calls,
                'Call_Rate': call_rate,
                'Failed_Samples': ', '.join(failed_samples) if failed_samples else 'None'
            })
        
        self.marker_call_rates = pd.DataFrame(results)
        self.marker_call_rates = self.marker_call_rates.sort_values('Call_Rate', ascending=False)
        
        print(f"  Calculated call rates for {len(self.marker_call_rates)} markers")
        print(f"  Call rate range: {self.marker_call_rates['Call_Rate'].min():.2%} - {self.marker_call_rates['Call_Rate'].max():.2%}")
        
    def filter_samples_by_call_rate(self):
        """Filter samples based on call rate threshold"""
        print(f"\nFiltering samples by call rate threshold ({self.sample_call_rate_threshold:.0%})...")
        
        self.passed_samples = self.sample_call_rates[
            self.sample_call_rates['Call_Rate'] >= self.sample_call_rate_threshold
        ].copy()
        
        self.failed_samples = self.sample_call_rates[
            self.sample_call_rates['Call_Rate'] < self.sample_call_rate_threshold
        ].copy()
        
        print(f"  Passed samples: {len(self.passed_samples)}")
        print(f"  Failed samples: {len(self.failed_samples)}")
        
    def calculate_b_allele_frequencies(self):
        """Calculate B allele frequencies for samples and markers"""
        print("\nCalculating B allele frequencies...")
        
        # Get data for passed samples only
        passed_sample_names = set(self.passed_samples['Sample'])
        passed_data = self.filtered_data[self.filtered_data['Sample'].isin(passed_sample_names)].copy()
        
        # Filter out NC calls for frequency calculations
        valid_data = passed_data[
            (passed_data['Genotype'] != 'NC') & 
            (passed_data['Genotype'].notna())
        ].copy()
        
        # Calculate B allele frequency for each sample
        sample_results = []
        for sample in passed_sample_names:
            sample_data = valid_data[valid_data['Sample'] == sample]
            
            aa_count = len(sample_data[sample_data['Genotype'] == 'AA'])
            ab_count = len(sample_data[sample_data['Genotype'] == 'AB'])
            bb_count = len(sample_data[sample_data['Genotype'] == 'BB'])
            
            total_alleles = (aa_count * 2) + (ab_count * 2) + (bb_count * 2)
            b_alleles = (ab_count) + (bb_count * 2)
            
            b_freq = b_alleles / total_alleles if total_alleles > 0 else 0
            
            sample_results.append({
                'Sample': sample,
                'AA_Count': aa_count,
                'AB_Count': ab_count,
                'BB_Count': bb_count,
                'Total_Calls': aa_count + ab_count + bb_count,
                'B_Allele_Frequency': b_freq
            })
        
        self.sample_b_freq = pd.DataFrame(sample_results)
        self.sample_b_freq = self.sample_b_freq.sort_values('B_Allele_Frequency', ascending=False)
        
        # Calculate B allele frequency for each marker
        marker_results = []
        markers = self.markerkey['Marker'].unique()
        
        for marker in markers:
            marker_data = valid_data[valid_data['Marker'] == marker]
            
            aa_count = len(marker_data[marker_data['Genotype'] == 'AA'])
            ab_count = len(marker_data[marker_data['Genotype'] == 'AB'])
            bb_count = len(marker_data[marker_data['Genotype'] == 'BB'])
            
            total_alleles = (aa_count * 2) + (ab_count * 2) + (bb_count * 2)
            b_alleles = (ab_count) + (bb_count * 2)
            
            b_freq = b_alleles / total_alleles if total_alleles > 0 else 0
            
            marker_results.append({
                'Marker': marker,
                'AA_Count': aa_count,
                'AB_Count': ab_count,
                'BB_Count': bb_count,
                'Total_Calls': aa_count + ab_count + bb_count,
                'B_Allele_Frequency': b_freq
            })
        
        self.marker_b_freq = pd.DataFrame(marker_results)
        self.marker_b_freq = self.marker_b_freq.sort_values('B_Allele_Frequency', ascending=False)
        
        print(f"  Calculated B allele frequencies for {len(self.sample_b_freq)} samples")
        print(f"  Calculated B allele frequencies for {len(self.marker_b_freq)} markers")
        
    def calculate_sample_concordance(self):
        """Calculate concordance between replicate samples"""
        print("\nCalculating sample concordance for replicates...")
        
        # Get passed samples only
        passed_sample_names = set(self.passed_samples['Sample'])
        passed_data = self.filtered_data[self.filtered_data['Sample'].isin(passed_sample_names)].copy()
        
        # Find replicate groups (samples with same prefix before underscore or dash)
        sample_groups = {}
        for sample in passed_sample_names:
            # Extract base name (everything before last underscore or dash followed by a number)
            base = sample.rsplit('_', 1)[0] if '_' in sample else sample.rsplit('-', 1)[0] if '-' in sample else sample
            if base not in sample_groups:
                sample_groups[base] = []
            sample_groups[base].append(sample)
        
        # Filter to only groups with 2+ samples
        replicate_groups = {k: v for k, v in sample_groups.items() if len(v) >= 2}
        
        if not replicate_groups:
            print("  No replicate groups found")
            return
        
        print(f"  Found {len(replicate_groups)} replicate groups")
        
        # Calculate concordance for each pair within groups
        concordance_results_passing = []
        concordance_results_with_failed = []
        
        for group_name, samples in replicate_groups.items():
            # Compare all pairs in the group
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    sample1 = samples[i]
                    sample2 = samples[j]
                    
                    # Get data for both samples
                    s1_data = passed_data[passed_data['Sample'] == sample1].set_index('Marker')
                    s2_data = passed_data[passed_data['Sample'] == sample2].set_index('Marker')
                    
                    # Find common markers
                    common_markers = list(set(s1_data.index) & set(s2_data.index))
                    
                    # PASSING DATA ANALYSIS (exclude NC)
                    concordant = 0
                    discordant = 0
                    discordant_details = []
                    
                    for marker in common_markers:
                        g1 = s1_data.loc[marker, 'Genotype']
                        g2 = s2_data.loc[marker, 'Genotype']
                        
                        # Skip if either is NC
                        if g1 == 'NC' or g2 == 'NC':
                            continue
                        
                        if g1 == g2:
                            concordant += 1
                        else:
                            discordant += 1
                            discordant_details.append(f"{marker}({g1}/{g2})")
                    
                    total_passing = concordant + discordant
                    concordance_rate_passing = concordant / total_passing if total_passing > 0 else 0
                    
                    concordance_results_passing.append({
                        'Group': group_name,
                        'Sample_1': sample1,
                        'Sample_2': sample2,
                        'Total_Comparisons': total_passing,
                        'Concordant': concordant,
                        'Discordant': discordant,
                        'Concordance_Rate': concordance_rate_passing,
                        'Discordant_Calls': ', '.join(discordant_details) if discordant_details else 'None'
                    })
                    
                    # WITH FAILED DATA ANALYSIS (include NC)
                    concordant_wf = 0
                    discordant_wf = 0
                    discordant_details_wf = []
                    
                    for marker in common_markers:
                        g1 = s1_data.loc[marker, 'Genotype']
                        g2 = s2_data.loc[marker, 'Genotype']
                        
                        if g1 == g2:
                            concordant_wf += 1
                        else:
                            discordant_wf += 1
                            discordant_details_wf.append(f"{marker}({g1}/{g2})")
                    
                    total_with_failed = concordant_wf + discordant_wf
                    concordance_rate_wf = concordant_wf / total_with_failed if total_with_failed > 0 else 0
                    
                    concordance_results_with_failed.append({
                        'Group': group_name,
                        'Sample_1': sample1,
                        'Sample_2': sample2,
                        'Total_Comparisons': total_with_failed,
                        'Concordant': concordant_wf,
                        'Discordant': discordant_wf,
                        'Concordance_Rate': concordance_rate_wf,
                        'Discordant_Calls': ', '.join(discordant_details_wf) if discordant_details_wf else 'None'
                    })
        
        self.concordance_passing = pd.DataFrame(concordance_results_passing)
        self.concordance_with_failed = pd.DataFrame(concordance_results_with_failed)
        
        if len(self.concordance_passing) > 0:
            print(f"  Calculated concordance for {len(self.concordance_passing)} sample pairs")
            print(f"  Passing data concordance range: {self.concordance_passing['Concordance_Rate'].min():.2%} - {self.concordance_passing['Concordance_Rate'].max():.2%}")
        
    def calculate_custom_sample_concordance(self):
        """Calculate concordance for custom sample groups"""
        print("\nCalculating custom sample concordance...")
        
        try:
            with open(self.custom_sample_groups, 'r') as f:
                lines = f.readlines()
            
            # Parse custom groups
            custom_groups = {}
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                samples = line.split()
                if len(samples) >= 2:
                    group_name = f"Group_{i+1}"
                    custom_groups[group_name] = samples
            
            if not custom_groups:
                print("  No valid custom sample groups found")
                return
            
            print(f"  Found {len(custom_groups)} custom sample groups")
            
            # Get passed samples only
            passed_sample_names = set(self.passed_samples['Sample'])
            passed_data = self.filtered_data[self.filtered_data['Sample'].isin(passed_sample_names)].copy()
            
            # Calculate concordance for each group
            concordance_results_passing = []
            concordance_results_with_failed = []
            
            for group_name, samples in custom_groups.items():
                # Filter to only samples that exist and passed
                valid_samples = [s for s in samples if s in passed_sample_names]
                
                if len(valid_samples) < 2:
                    print(f"  Warning: {group_name} has fewer than 2 valid samples, skipping")
                    continue
                
                # Compare all pairs in the group
                for i in range(len(valid_samples)):
                    for j in range(i + 1, len(valid_samples)):
                        sample1 = valid_samples[i]
                        sample2 = valid_samples[j]
                        
                        # Get data for both samples
                        s1_data = passed_data[passed_data['Sample'] == sample1].set_index('Marker')
                        s2_data = passed_data[passed_data['Sample'] == sample2].set_index('Marker')
                        
                        # Find common markers
                        common_markers = list(set(s1_data.index) & set(s2_data.index))
                        
                        # PASSING DATA ANALYSIS (exclude NC)
                        concordant = 0
                        discordant = 0
                        discordant_details = []
                        
                        for marker in common_markers:
                            g1 = s1_data.loc[marker, 'Genotype']
                            g2 = s2_data.loc[marker, 'Genotype']
                            
                            # Skip if either is NC
                            if g1 == 'NC' or g2 == 'NC':
                                continue
                            
                            if g1 == g2:
                                concordant += 1
                            else:
                                discordant += 1
                                discordant_details.append(f"{marker}({g1}/{g2})")
                        
                        total_passing = concordant + discordant
                        concordance_rate_passing = concordant / total_passing if total_passing > 0 else 0
                        
                        concordance_results_passing.append({
                            'Group': group_name,
                            'Sample_1': sample1,
                            'Sample_2': sample2,
                            'Total_Comparisons': total_passing,
                            'Concordant': concordant,
                            'Discordant': discordant,
                            'Concordance_Rate': concordance_rate_passing,
                            'Discordant_Calls': ', '.join(discordant_details) if discordant_details else 'None'
                        })
                        
                        # WITH FAILED DATA ANALYSIS (include NC)
                        concordant_wf = 0
                        discordant_wf = 0
                        discordant_details_wf = []
                        
                        for marker in common_markers:
                            g1 = s1_data.loc[marker, 'Genotype']
                            g2 = s2_data.loc[marker, 'Genotype']
                            
                            if g1 == g2:
                                concordant_wf += 1
                            else:
                                discordant_wf += 1
                                discordant_details_wf.append(f"{marker}({g1}/{g2})")
                        
                        total_with_failed = concordant_wf + discordant_wf
                        concordance_rate_wf = concordant_wf / total_with_failed if total_with_failed > 0 else 0
                        
                        concordance_results_with_failed.append({
                            'Group': group_name,
                            'Sample_1': sample1,
                            'Sample_2': sample2,
                            'Total_Comparisons': total_with_failed,
                            'Concordant': concordant_wf,
                            'Discordant': discordant_wf,
                            'Concordance_Rate': concordance_rate_wf,
                            'Discordant_Calls': ', '.join(discordant_details_wf) if discordant_details_wf else 'None'
                        })
            
            self.custom_sample_concordance_passing = pd.DataFrame(concordance_results_passing)
            self.custom_sample_concordance_with_failed = pd.DataFrame(concordance_results_with_failed)
            
            if len(concordance_results_passing) > 0:
                print(f"  Calculated concordance for {len(concordance_results_passing)} custom sample pairs")
            
        except Exception as e:
            print(f"  Error calculating custom sample concordance: {e}")
    
    def calculate_custom_marker_concordance(self):
        """Calculate concordance for custom marker groups with G/T format conversion"""
        print("\nCalculating custom marker concordance...")
        
        try:
            with open(self.custom_marker_groups, 'r') as f:
                lines = f.readlines()
            
            # Parse custom marker groups
            custom_groups = {}
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                markers = line.split()
                if len(markers) >= 2:
                    group_name = f"Group_{i+1}"
                    custom_groups[group_name] = markers
            
            if not custom_groups:
                print("  No valid custom marker groups found")
                return
            
            print(f"  Found {len(custom_groups)} custom marker groups")
            
            # Get passed samples only
            passed_sample_names = set(self.passed_samples['Sample'])
            passed_data = self.filtered_data[self.filtered_data['Sample'].isin(passed_sample_names)].copy()
            
            # Calculate concordance for each group
            concordance_results_passing = []
            concordance_results_with_failed = []
            
            for group_name, markers in custom_groups.items():
                # Filter to only markers that exist in the key
                valid_markers = [m for m in markers if m in set(self.markerkey['Marker'])]
                
                if len(valid_markers) < 2:
                    print(f"  Warning: {group_name} has fewer than 2 valid markers, skipping")
                    continue
                
                # Compare all pairs in the group
                for i in range(len(valid_markers)):
                    for j in range(i + 1, len(valid_markers)):
                        marker1 = valid_markers[i]
                        marker2 = valid_markers[j]
                        
                        # Get data for both markers
                        m1_data = passed_data[passed_data['Marker'] == marker1].set_index('Sample')
                        m2_data = passed_data[passed_data['Marker'] == marker2].set_index('Sample')
                        
                        # Find common samples
                        common_samples = list(set(m1_data.index) & set(m2_data.index))
                        
                        # PASSING DATA ANALYSIS (exclude NC, convert to G/T format)
                        concordant = 0
                        discordant = 0
                        discordant_details = []
                        
                        for sample in common_samples:
                            g1_ab = m1_data.loc[sample, 'Genotype']
                            g2_ab = m2_data.loc[sample, 'Genotype']
                            
                            # Skip if either is NC
                            if g1_ab == 'NC' or g2_ab == 'NC':
                                continue
                            
                            # Convert to G/T format
                            g1_gt = self.convert_genotype_to_alleles(marker1, g1_ab)
                            g2_gt = self.convert_genotype_to_alleles(marker2, g2_ab)
                            
                            # Skip if conversion failed
                            if g1_gt == 'NC' or g2_gt == 'NC':
                                continue
                            
                            if g1_gt == g2_gt:
                                concordant += 1
                            else:
                                discordant += 1
                                discordant_details.append(f"{sample}({g1_gt}/{g2_gt})")
                        
                        total_passing = concordant + discordant
                        concordance_rate_passing = concordant / total_passing if total_passing > 0 else 0
                        
                        concordance_results_passing.append({
                            'Group': group_name,
                            'Marker_1': marker1,
                            'Marker_2': marker2,
                            'Total_Comparisons': total_passing,
                            'Concordant': concordant,
                            'Discordant': discordant,
                            'Concordance_Rate': concordance_rate_passing,
                            'Discordant_Calls': ', '.join(discordant_details) if discordant_details else 'None'
                        })
                        
                        # WITH FAILED DATA ANALYSIS (include NC, convert to G/T format)
                        concordant_wf = 0
                        discordant_wf = 0
                        discordant_details_wf = []
                        
                        for sample in common_samples:
                            g1_ab = m1_data.loc[sample, 'Genotype']
                            g2_ab = m2_data.loc[sample, 'Genotype']
                            
                            # Convert to G/T format
                            g1_gt = self.convert_genotype_to_alleles(marker1, g1_ab)
                            g2_gt = self.convert_genotype_to_alleles(marker2, g2_ab)
                            
                            if g1_gt == g2_gt:
                                concordant_wf += 1
                            else:
                                discordant_wf += 1
                                discordant_details_wf.append(f"{sample}({g1_gt}/{g2_gt})")
                        
                        total_with_failed = concordant_wf + discordant_wf
                        concordance_rate_wf = concordant_wf / total_with_failed if total_with_failed > 0 else 0
                        
                        concordance_results_with_failed.append({
                            'Group': group_name,
                            'Marker_1': marker1,
                            'Marker_2': marker2,
                            'Total_Comparisons': total_with_failed,
                            'Concordant': concordant_wf,
                            'Discordant': discordant_wf,
                            'Concordance_Rate': concordance_rate_wf,
                            'Discordant_Calls': ', '.join(discordant_details_wf) if discordant_details_wf else 'None'
                        })
            
            self.custom_marker_concordance_passing = pd.DataFrame(concordance_results_passing)
            self.custom_marker_concordance_with_failed = pd.DataFrame(concordance_results_with_failed)
            
            if len(concordance_results_passing) > 0:
                print(f"  Calculated concordance for {len(concordance_results_passing)} custom marker pairs")
                print(f"  Passing data concordance range: {self.custom_marker_concordance_passing['Concordance_Rate'].min():.2%} - {self.custom_marker_concordance_passing['Concordance_Rate'].max():.2%}")
            
        except Exception as e:
            print(f"  Error calculating custom marker concordance: {e}")
            import traceback
            traceback.print_exc()
    
    def create_plots(self):
        """Create all visualizations"""
        print("\nCreating visualizations...")
        plots = {}
        
        # Sample call rate distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=self.sample_call_rates['Call_Rate'] * 100,
            nbinsx=30,
            marker_color='steelblue',
            name='Sample Call Rate'
        ))
        fig.add_vline(
            x=self.sample_call_rate_threshold * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {self.sample_call_rate_threshold:.0%}"
        )
        fig.update_layout(
            title="Sample Call Rate Distribution",
            xaxis_title="Call Rate (%)",
            yaxis_title="Number of Samples",
            height=400
        )
        plots['sample_call_rate_dist'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Marker call rate distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=self.marker_call_rates['Call_Rate'] * 100,
            nbinsx=30,
            marker_color='coral',
            name='Marker Call Rate'
        ))
        fig.update_layout(
            title="Marker Call Rate Distribution",
            xaxis_title="Call Rate (%)",
            yaxis_title="Number of Markers",
            height=400
        )
        plots['marker_call_rate_dist'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Sample B allele frequency distribution
        if self.sample_b_freq is not None and len(self.sample_b_freq) > 0:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=self.sample_b_freq['B_Allele_Frequency'] * 100,
                nbinsx=30,
                marker_color='green',
                name='B Allele Frequency'
            ))
            fig.update_layout(
                title="Sample B Allele Frequency Distribution",
                xaxis_title="B Allele Frequency (%)",
                yaxis_title="Number of Samples",
                height=400
            )
            plots['sample_b_freq_dist'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Marker B allele frequency distribution
        if self.marker_b_freq is not None and len(self.marker_b_freq) > 0:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=self.marker_b_freq['B_Allele_Frequency'] * 100,
                nbinsx=30,
                marker_color='purple',
                name='B Allele Frequency'
            ))
            fig.update_layout(
                title="Marker B Allele Frequency Distribution",
                xaxis_title="B Allele Frequency (%)",
                yaxis_title="Number of Markers",
                height=400
            )
            plots['marker_b_freq_dist'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Sample concordance plots
        if self.concordance_passing is not None and len(self.concordance_passing) > 0:
            # Concordance rate plot with numbered pairs and hover info
            comparison_numbers = list(range(1, len(self.concordance_passing) + 1))
            hover_text = [f"Pair {i}<br>{row['Sample_1']}<br>vs<br>{row['Sample_2']}<br>Concordance: {row['Concordance_Rate']*100:.2f}%" 
                         for i, (_, row) in enumerate(self.concordance_passing.iterrows(), 1)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comparison_numbers,
                y=self.concordance_passing['Concordance_Rate'] * 100,
                marker_color=['green' if x >= 95 else 'orange' if x >= 90 else 'firebrick' 
                             for x in self.concordance_passing['Concordance_Rate'] * 100],
                text=[f"{x:.2f}%" for x in self.concordance_passing['Concordance_Rate'] * 100],
                textposition='outside',
                hovertext=hover_text,
                hoverinfo='text'
            ))
            fig.update_layout(
                title="Sample Concordance Rates (Passing Data)",
                xaxis_title="Comparison Pair Number",
                yaxis_title="Concordance Rate (%)",
                height=500,
                showlegend=False
            )
            plots['sample_concordance'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            # Discordant calls by marker
            marker_discordance = {}
            for _, row in self.concordance_passing.iterrows():
                if row['Discordant_Calls'] != 'None':
                    for call in row['Discordant_Calls'].split(', '):
                        if '(' in call:
                            marker = call.split('(')[0]
                            marker_discordance[marker] = marker_discordance.get(marker, 0) + 1
            
            if marker_discordance:
                markers_sorted = sorted(marker_discordance.items(), key=lambda x: x[1], reverse=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[m[0] for m in markers_sorted[:20]],
                    y=[m[1] for m in markers_sorted[:20]],
                    marker_color='firebrick'
                ))
                fig.update_layout(
                    title="Top 20 Markers with Discordant Calls (Sample Concordance)",
                    xaxis_title="Marker",
                    yaxis_title="Number of Discordant Calls",
                    height=500,
                    xaxis_tickangle=-45
                )
                plots['replicate_marker_discordance'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Custom sample concordance plots
        if self.custom_sample_concordance_passing is not None and len(self.custom_sample_concordance_passing) > 0:
            # Concordance rate plot with numbered pairs and hover info
            comparison_numbers = list(range(1, len(self.custom_sample_concordance_passing) + 1))
            hover_text = [f"Pair {i}<br>{row['Sample_1']}<br>vs<br>{row['Sample_2']}<br>Concordance: {row['Concordance_Rate']*100:.2f}%" 
                         for i, (_, row) in enumerate(self.custom_sample_concordance_passing.iterrows(), 1)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comparison_numbers,
                y=self.custom_sample_concordance_passing['Concordance_Rate'] * 100,
                marker_color=['green' if x >= 95 else 'orange' if x >= 90 else 'firebrick' 
                             for x in self.custom_sample_concordance_passing['Concordance_Rate'] * 100],
                text=[f"{x:.2f}%" for x in self.custom_sample_concordance_passing['Concordance_Rate'] * 100],
                textposition='outside',
                hovertext=hover_text,
                hoverinfo='text'
            ))
            fig.update_layout(
                title="Custom Sample Concordance Rates (Passing Data)",
                xaxis_title="Comparison Pair Number",
                yaxis_title="Concordance Rate (%)",
                height=500,
                showlegend=False
            )
            plots['custom_sample_concordance'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            # Discordant calls by marker
            marker_discordance = {}
            for _, row in self.custom_sample_concordance_passing.iterrows():
                if row['Discordant_Calls'] != 'None':
                    for call in row['Discordant_Calls'].split(', '):
                        if '(' in call:
                            marker = call.split('(')[0]
                            marker_discordance[marker] = marker_discordance.get(marker, 0) + 1
            
            if marker_discordance:
                markers_sorted = sorted(marker_discordance.items(), key=lambda x: x[1], reverse=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[m[0] for m in markers_sorted[:20]],
                    y=[m[1] for m in markers_sorted[:20]],
                    marker_color='firebrick'
                ))
                fig.update_layout(
                    title="Top 20 Markers with Discordant Calls (Custom Sample Concordance)",
                    xaxis_title="Marker",
                    yaxis_title="Number of Discordant Calls",
                    height=500,
                    xaxis_tickangle=-45
                )
                plots['custom_sample_marker_discordance'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Custom marker concordance plots
        if self.custom_marker_concordance_passing is not None and len(self.custom_marker_concordance_passing) > 0:
            # Concordance rate plot with numbered pairs and hover info showing actual marker names
            comparison_numbers = list(range(1, len(self.custom_marker_concordance_passing) + 1))
            hover_text = [f"Pair {i}<br>Group: {row['Group']}<br>{row['Marker_1']}<br>vs<br>{row['Marker_2']}<br>Concordance: {row['Concordance_Rate']*100:.2f}%" 
                         for i, (_, row) in enumerate(self.custom_marker_concordance_passing.iterrows(), 1)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comparison_numbers,
                y=self.custom_marker_concordance_passing['Concordance_Rate'] * 100,
                marker_color=['green' if x >= 95 else 'orange' if x >= 90 else 'firebrick' 
                             for x in self.custom_marker_concordance_passing['Concordance_Rate'] * 100],
                text=[f"{x:.2f}%" for x in self.custom_marker_concordance_passing['Concordance_Rate'] * 100],
                textposition='outside',
                hovertext=hover_text,
                hoverinfo='text'
            ))
            fig.update_layout(
                title="Custom Marker Concordance Rates (Passing Data, G/T Format)",
                xaxis_title="Comparison Pair Number",
                yaxis_title="Concordance Rate (%)",
                height=500,
                showlegend=False
            )
            plots['custom_marker_concordance'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            # Discordant calls by sample
            sample_discordance = {}
            for _, row in self.custom_marker_concordance_passing.iterrows():
                if row['Discordant_Calls'] != 'None':
                    for call in row['Discordant_Calls'].split(', '):
                        if '(' in call:
                            sample = call.split('(')[0]
                            sample_discordance[sample] = sample_discordance.get(sample, 0) + 1
            
            if sample_discordance:
                samples_sorted = sorted(sample_discordance.items(), key=lambda x: x[1], reverse=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[s[0] for s in samples_sorted[:20]],
                    y=[s[1] for s in samples_sorted[:20]],
                    marker_color='firebrick'
                ))
                fig.update_layout(
                    title="Top 20 Samples with Discordant Calls (Custom Marker Concordance)",
                    xaxis_title="Sample",
                    yaxis_title="Number of Discordant Calls",
                    height=500,
                    xaxis_tickangle=-45
                )
                plots['custom_marker_sample_discordance'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        print("  Created all visualizations")
        return plots
    
    def save_output_files(self):
        """Save all output CSV files"""
        print("\nSaving output files...")
        output_files = {}
        
        # Sample call rates
        file_path = self.output_dir / f"{self.report_name}_SampleCallRates.csv"
        self.sample_call_rates.to_csv(file_path, index=False)
        output_files['sample_call_rates'] = file_path
        print(f"  Saved: {file_path}")
        
        # Marker call rates
        file_path = self.output_dir / f"{self.report_name}_MarkerCallRates.csv"
        self.marker_call_rates.to_csv(file_path, index=False)
        output_files['marker_call_rates'] = file_path
        print(f"  Saved: {file_path}")
        
        # Passed samples
        file_path = self.output_dir / f"{self.report_name}_PassedSamples.csv"
        self.passed_samples.to_csv(file_path, index=False)
        output_files['passed_samples'] = file_path
        print(f"  Saved: {file_path}")
        
        # Failed samples
        file_path = self.output_dir / f"{self.report_name}_FailedSamples.csv"
        self.failed_samples.to_csv(file_path, index=False)
        output_files['failed_samples'] = file_path
        print(f"  Saved: {file_path}")
        
        # Sample B allele frequencies
        if self.sample_b_freq is not None and len(self.sample_b_freq) > 0:
            file_path = self.output_dir / f"{self.report_name}_SampleBAlleleFrequencies.csv"
            self.sample_b_freq.to_csv(file_path, index=False)
            output_files['sample_b_freq'] = file_path
            print(f"  Saved: {file_path}")
        
        # Marker B allele frequencies
        if self.marker_b_freq is not None and len(self.marker_b_freq) > 0:
            file_path = self.output_dir / f"{self.report_name}_MarkerBAlleleFrequencies.csv"
            self.marker_b_freq.to_csv(file_path, index=False)
            output_files['marker_b_freq'] = file_path
            print(f"  Saved: {file_path}")
        
        # Passed samples genotypes (converted to G/T format, no NC)
        passed_sample_names = set(self.passed_samples['Sample'])
        passed_data = self.filtered_data[self.filtered_data['Sample'].isin(passed_sample_names)].copy()
        
        # Filter out NC genotypes
        passed_data_no_nc = passed_data[
            (passed_data['Genotype'] != 'NC') & 
            (passed_data['Genotype'].notna())
        ].copy()
        
        # Convert genotypes to G/T format
        passed_data_no_nc['Genotype_GT'] = passed_data_no_nc.apply(
            lambda row: self.convert_genotype_to_alleles(row['Marker'], row['Genotype']),
            axis=1
        )
        
        # Create output with only G/T format
        passed_genotypes_output = passed_data_no_nc[['Marker', 'Sample', 'Genotype_GT']].copy()
        passed_genotypes_output.columns = ['Marker', 'Sample', 'Genotype']
        
        file_path = self.output_dir / f"{self.report_name}_PassedSamplesGenotypes_GT.csv"
        passed_genotypes_output.to_csv(file_path, index=False)
        output_files['passed_genotypes_gt'] = file_path
        print(f"  Saved: {file_path}")
        
        # Sample concordance
        if self.concordance_passing is not None and len(self.concordance_passing) > 0:
            file_path = self.output_dir / f"{self.report_name}_SampleConcordance_Passing.csv"
            self.concordance_passing.to_csv(file_path, index=False)
            output_files['concordance_passing'] = file_path
            print(f"  Saved: {file_path}")
            
            file_path = self.output_dir / f"{self.report_name}_SampleConcordance_WithFailed.csv"
            self.concordance_with_failed.to_csv(file_path, index=False)
            output_files['concordance_with_failed'] = file_path
            print(f"  Saved: {file_path}")
        
        # Custom sample concordance
        if self.custom_sample_concordance_passing is not None and len(self.custom_sample_concordance_passing) > 0:
            file_path = self.output_dir / f"{self.report_name}_CustomSampleConcordance_Passing.csv"
            self.custom_sample_concordance_passing.to_csv(file_path, index=False)
            output_files['custom_sample_concordance_passing'] = file_path
            print(f"  Saved: {file_path}")
            
            file_path = self.output_dir / f"{self.report_name}_CustomSampleConcordance_WithFailed.csv"
            self.custom_sample_concordance_with_failed.to_csv(file_path, index=False)
            output_files['custom_sample_concordance_with_failed'] = file_path
            print(f"  Saved: {file_path}")
        
        # Custom marker concordance
        if self.custom_marker_concordance_passing is not None and len(self.custom_marker_concordance_passing) > 0:
            file_path = self.output_dir / f"{self.report_name}_CustomMarkerConcordance_Passing_GT.csv"
            self.custom_marker_concordance_passing.to_csv(file_path, index=False)
            output_files['custom_marker_concordance_passing'] = file_path
            print(f"  Saved: {file_path}")
            
            file_path = self.output_dir / f"{self.report_name}_CustomMarkerConcordance_WithFailed_GT.csv"
            self.custom_marker_concordance_with_failed.to_csv(file_path, index=False)
            output_files['custom_marker_concordance_with_failed'] = file_path
            print(f"  Saved: {file_path}")
        
        return output_files
    
    def generate_html_report(self, plots, output_files):
        """Generate comprehensive HTML report"""
        print("\nGenerating HTML report...")
        
        report_file = self.output_dir / f"{self.report_name}_Report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.report_name} - ViA 2</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1f618d;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: #c3c3c1;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #c22e2d;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #c22e2d;
            padding-left: 15px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-item {{
            margin: 10px 0;
            font-size: 16px;
        }}
        .summary-label {{
            font-weight: bold;
            color: #1f618d;
        }}
        .plot-container {{
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            background-color: #1f618d;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .data-table th {{
            background-color: #1f618d;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        .data-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .data-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .file-link {{
            display: inline-block;
            margin: 5px 10px 5px 0;
            padding: 8px 15px;
            background-color: #1f618d;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 14px;
        }}
        .file-link:hover {{
            background-color: #2980b9;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>VGL iScan Analysis (VIA 2): {self.report_name}</h1>
        
        <div class="summary-box">
            <h2>Analysis Summary</h2>
   <div class="summary-item"><span class="summary-label">Analysis Date:</span> {self.timestamp}</div>
            <div class="summary-item"><span class="summary-label">Program Version:</span> 2.5</div>
            <div class="summary-item"><span class="summary-label">Command:</span> <code style="background-color: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 13px;">{self.command_line if self.command_line else 'N/A'}</code></div>
            <div class="summary-item"><span class="summary-label">Total Samples:</span> {len(self.sample_call_rates)}</div>
            <div class="summary-item"><span class="summary-label">Passed Samples:</span> {len(self.passed_samples)}</div>
            <div class="summary-item"><span class="summary-label">Failed Samples:</span> {len(self.failed_samples)}</div>
            <div class="summary-item"><span class="summary-label">Total Markers:</span> {len(self.marker_call_rates)}</div>
            <div class="summary-item"><span class="summary-label">Call Rate Threshold:</span> {self.sample_call_rate_threshold:.0%}</div>
            <div class="summary-item"><span class="summary-label">Exclude Failed Markers:</span> {'Yes' if self.exclude_failed_markers else 'No'}</div>
        </div>
        
        <h2>Output Files</h2>
        <div style="margin: 20px 0;">
"""
        
        for name, path in output_files.items():
            html_content += f'            <a href="{path.name}" class="file-link" download>{path.name}</a>\n'
        
        html_content += """
        </div>
        
        <h2>Sample Call Rates</h2>
        <div class="plot-container">
"""
        html_content += plots['sample_call_rate_dist']
        html_content += """
        </div>
        
        <h2>Marker Call Rates</h2>
        <div class="plot-container">
"""
        html_content += plots['marker_call_rate_dist']
        html_content += """
        </div>
"""
        
        # Add B allele frequency plots if available
        if 'sample_b_freq_dist' in plots:
            html_content += """
        <h2>Sample B Allele Frequencies</h2>
        <div class="plot-container">
"""
            html_content += plots['sample_b_freq_dist']
            html_content += """
        </div>
"""
        
        if 'marker_b_freq_dist' in plots:
            html_content += """
        <h2>Marker B Allele Frequencies</h2>
        <div class="plot-container">
"""
            html_content += plots['marker_b_freq_dist']
            html_content += """
        </div>
"""
        
        # Add sample concordance section
        if self.concordance_passing is not None and len(self.concordance_passing) > 0:
            html_content += """
        <h2>Sample Concordance Analysis (Replicates)</h2>
        <p>Analysis of concordance between replicate samples</p>
        
        <h3>Replicate Sample Concordance (Passing Data)</h3>
        <div class="plot-container">
"""
            if 'sample_concordance' in plots:
                html_content += plots['sample_concordance']
            
            html_content += """
        </div>
        
        <table class="data-table">
            <thead>
                <tr>
                    <th>Group</th>
                    <th>Sample 1</th>
                    <th>Sample 2</th>
                    <th>Total Comparisons</th>
                    <th>Concordant</th>
                    <th>Discordant</th>
                    <th>Concordance Rate</th>
                    <th>Discordant Calls</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for _, row in self.concordance_passing.iterrows():
                status_color = 'green' if row['Concordance_Rate'] >= 0.95 else 'orange' if row['Concordance_Rate'] >= 0.90 else 'firebrick'
                html_content += f"""
                <tr>
                    <td><strong>{row['Group']}</strong></td>
                    <td>{row['Sample_1']}</td>
                    <td>{row['Sample_2']}</td>
                    <td>{row['Total_Comparisons']}</td>
                    <td style="color: green;">{row['Concordant']}</td>
                    <td style="color: firebrick;">{row['Discordant']}</td>
                    <td style="color: {status_color}; font-weight: bold;">{row['Concordance_Rate']:.2%}</td>
                    <td style="font-size: 11px;">{row['Discordant_Calls']}</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
"""
            
            # Add marker discordance plot if available
            if 'replicate_marker_discordance' in plots:
                html_content += """
        <h3>Discordant Calls by Marker (Replicate Samples)</h3>
        <div class="plot-container">
"""
                html_content += plots['replicate_marker_discordance']
                html_content += """
        </div>
"""
        
        # Add custom sample concordance section
        if self.custom_sample_concordance_passing is not None and len(self.custom_sample_concordance_passing) > 0:
            html_content += """
        <h2>Custom Sample Concordance Analysis</h2>
        <p>Concordance for user-specified sample groups</p>
        
        <h3>Custom Sample Concordance (Passing Data)</h3>
        <div class="plot-container">
"""
            if 'custom_sample_concordance' in plots:
                html_content += plots['custom_sample_concordance']
            
            html_content += """
        </div>
        
        <table class="data-table">
            <thead>
                <tr>
                    <th>Group</th>
                    <th>Sample 1</th>
                    <th>Sample 2</th>
                    <th>Total Comparisons</th>
                    <th>Concordant</th>
                    <th>Discordant</th>
                    <th>Concordance Rate</th>
                    <th>Discordant Calls</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for _, row in self.custom_sample_concordance_passing.iterrows():
                status_color = 'green' if row['Concordance_Rate'] >= 0.95 else 'orange' if row['Concordance_Rate'] >= 0.90 else 'firebrick'
                html_content += f"""
                <tr>
                    <td><strong>{row['Group']}</strong></td>
                    <td>{row['Sample_1']}</td>
                    <td>{row['Sample_2']}</td>
                    <td>{row['Total_Comparisons']}</td>
                    <td style="color: green;">{row['Concordant']}</td>
                    <td style="color: firebrick;">{row['Discordant']}</td>
                    <td style="color: {status_color}; font-weight: bold;">{row['Concordance_Rate']:.2%}</td>
                    <td style="font-size: 11px;">{row['Discordant_Calls']}</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
"""
            
            # Add marker discordance plot for custom samples if available
            if 'custom_sample_marker_discordance' in plots:
                html_content += """
        <h3>Discordant Calls by Marker (Custom Samples)</h3>
        <div class="plot-container">
"""
                html_content += plots['custom_sample_marker_discordance']
                html_content += """
        </div>
"""
        
        # Add custom marker concordance section
        if self.custom_marker_concordance_passing is not None and len(self.custom_marker_concordance_passing) > 0:
            html_content += """
        <h2>Custom Marker Concordance Analysis</h2>
        <p>Concordance for user-specified marker groups (G/T format)</p>
        
        <h3>Custom Marker Concordance (Passing Data)</h3>
        <div class="plot-container">
"""
            if 'custom_marker_concordance' in plots:
                html_content += plots['custom_marker_concordance']
            
            html_content += """
        </div>
        
        <table class="data-table">
            <thead>
                <tr>
                    <th>Group</th>
                    <th>Marker 1</th>
                    <th>Marker 2</th>
                    <th>Total Comparisons</th>
                    <th>Concordant</th>
                    <th>Discordant</th>
                    <th>Concordance Rate</th>
                    <th>Discordant Calls</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for _, row in self.custom_marker_concordance_passing.iterrows():
                status_color = 'green' if row['Concordance_Rate'] >= 0.95 else 'orange' if row['Concordance_Rate'] >= 0.90 else 'firebrick'
                html_content += f"""
                <tr>
                    <td><strong>{row['Group']}</strong></td>
                    <td>{row['Marker_1']}</td>
                    <td>{row['Marker_2']}</td>
                    <td>{row['Total_Comparisons']}</td>
                    <td style="color: green;">{row['Concordant']}</td>
                    <td style="color: firebrick;">{row['Discordant']}</td>
                    <td style="color: {status_color}; font-weight: bold;">{row['Concordance_Rate']:.2%}</td>
                    <td style="font-size: 11px;">{row['Discordant_Calls']}</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
"""
            
            # Add sample discordance plot for custom markers if available
            if 'custom_marker_sample_discordance' in plots:
                html_content += """
        <h3>Discordant Calls by Sample (Custom Markers)</h3>
        <div class="plot-container">
"""
                html_content += plots['custom_marker_sample_discordance']
                html_content += """
        </div>
"""
        
        html_content += """
        <div class="footer">
            <p>END OF REPORT</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"  Report saved: {report_file}")
        return report_file
        
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 70)
        print(f"GENOTYPE ANALYSIS: {self.report_name}")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Calculate call rates
        self.calculate_sample_call_rates()
        self.calculate_marker_call_rates()
        
        # Filter samples
        self.filter_samples_by_call_rate()
        
        # Calculate B allele frequencies
        self.calculate_b_allele_frequencies()
        
        # Calculate concordance if requested
        if self.run_concordance:
            self.calculate_sample_concordance()
        
        # Calculate custom sample concordance if file provided
        if self.custom_sample_groups:
            self.calculate_custom_sample_concordance()
        
        # Calculate custom marker concordance if file provided
        if self.custom_marker_groups:
            self.calculate_custom_marker_concordance()
        
        # Create visualizations
        plots = self.create_plots()
        
        # Save output files
        output_files = self.save_output_files()
        
        # Generate HTML report
        report_file = self.generate_html_report(plots, output_files)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\nHTML Report: {report_file}")
        print(f"\nAll output files saved to: {self.output_dir}")
        
        return report_file


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Genotype Data Analysis Program V2.5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python genotype_analyzerV2_5.py -r rawdata -k markerkey -n SampleAgeTest -c 90
  python genotype_analyzerV2_5.py -r rawdata -k markerkey -n SampleAgeTest -c 90 --exclude-failed-markers
  
Input file formats:
  Raw data: 3 columns - Marker, Sample, Genotype (AA/AB/BB/NC)
  Marker key: 3 columns - Marker, A_Allele, B_Allele (no header)
  
Version 2.5 Changes:
  - Custom marker concordance plot now uses numbered pairs with hover info showing marker names
  - Added --exclude-failed-markers option to exclude markers with 0%% call rate from sample call rate denominator
  - Improved readability of custom marker concordance visualizations
        """
    )
    
    parser.add_argument('-r', '--rawdata', required=True,
                        help='Path to raw genotype data file')
    parser.add_argument('-k', '--markerkey', required=True,
                        help='Path to marker key file')
    parser.add_argument('-n', '--name', required=True,
                        help='Report name (e.g., SampleAgeTest)')
    parser.add_argument('-c', '--callrate', required=True, type=int,
                        help='Minimum sample call rate threshold (0-100)')
    parser.add_argument('--concordance', action='store_true',
                        help='Run sample concordance analysis for replicate samples')
    parser.add_argument('--custom-samples', dest='custom_samples',
                        help='File with custom sample groups for concordance (one group per line)')
    parser.add_argument('--custom-markers', dest='custom_markers',
                        help='File with custom marker groups for concordance (one group per line)')
    parser.add_argument('--exclude-failed-markers', dest='exclude_failed_markers', action='store_true',
                        help='Exclude markers with 0%% call rate from sample call rate denominator')
    
    args = parser.parse_args()
    
    # Validate call rate
    if not 0 <= args.callrate <= 100:
        print("Error: Call rate must be between 0 and 100")
        sys.exit(1)
    
# Create analyzer and run
    analyzer = GenotypeAnalyzer(
        rawdata_file=args.rawdata,
        markerkey_file=args.markerkey,
        report_name=args.name,
        sample_call_rate_threshold=args.callrate,
        run_concordance=args.concordance,
        custom_sample_groups=args.custom_samples,
        custom_marker_groups=args.custom_markers,
        exclude_failed_markers=args.exclude_failed_markers
    )
    
    # Store the command line used
    import sys
    analyzer.command_line = ' '.join(sys.argv)
    
    try:
        analyzer.run_analysis()
    except Exception as e:
        print(f"\nâŒ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
