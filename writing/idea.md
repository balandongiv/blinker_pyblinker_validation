#  Executive Summary of Research Proposal

(Please include the problem statement, objectives, research methodology, expected output/outcomes/implication, and 
significance of
output from the research project)
This research proposal aims to address the challenge posed by eye-blink artifacts in electroencephalography (EEG) recordings—a
common yet complex issue in neuroscience. The presence of these artifacts offers an opportunity to understand the correlation
between eye blinks and neural activity changes during cognitive tasks, but their accurate identification and annotation have been
hampered by manual processing. This project introduces the development of MNE-Pyblinker, a Python-compatible version of the
existing MATLAB tool BLINKERS, designed to automate the detection of ocular indices, enhancing efficiency, and accessibility for
the broader neuroscience community.
The objectives of this research are twofold: firstly, to migrate BLINKERS to a Python environment, ensuring seamless integration
with MNE-Python; and secondly, to improve the tool's accuracy and generalizability across diverse datasets. This involves a
methodology structured into two phases: the migration and integration of BLINKERS into Python, followed by the enhancement of its
functionality and generalization capability. These phases include rigorous component migration, unit testing, validation, development
of a new preprocessing pipeline, and optimization of algorithm-free parameters.Expected outcomes of this project include a significant reduction in the time required for EEG data preprocessing, enhanced
                                                                               accuracy in eye-blink artifact detection, and the discovery of novel eye-based biomarkers for neurological conditions or cognitive
                                                                               states. This research aim to streamline EEG data analysis workflows, contribute to the field of neuroscience by facilitating more
                                                                               accurate diagnoses, and support the development of new EEG analysis tools with potential commercial applications.
                                                                               The significance of this project extends beyond the technical advancements. By improving EEG signal processing, it aligns with
                                                                               government health reform strategies focusing on digital technologies, supports environmental sustainability through optimized
                                                                               computational resources, and fosters academic and industrial innovation. Additionally, the application of such a tool in fatigue
                                                                               research will improve workplace safety and health, highlighting the project's broad relevance and potential to contribute to societal
                                                                               well-being.

#  Problem Statement
The presence of eye-blink artifacts in electroencephalography (EEG) recordings presents both an opportunity and a challenge in
neuroscience research. These artifacts, indicative of eye blinks, can be analyzed to understand their correlation with neural activity
changes during specific cognitive tasks. Yet, the accurate identification and annotation of these artifacts are marred by the need for
manual processing, a skill-intensive and time-consuming task that becomes increasingly laborious with larger datasets. To address
this issue, a package name BLINKERS has been developed to identify and extract the ocular indices in an automated fashion.
However, the BLINKERS is hampered by two significant limitations: parameter tuning relies on a limited dataset and employs a rigid
approach, leading to inconsistencies in results; and its availability solely in MATLAB restricts its accessibility to the broader
neuroscience community that predominantly utilizes Python for data analysis.
This gap underscores a strong motivation to migrate the MATLAB-version BLINKER to a Python-compatible environment under the
code name MNE-Pyblinker. Such migration aims not only to retain the sophisticated functionalities of the original BLINKERS but also
to extend its reach and efficacy by leveraging the widespread adoption of Python among researchers. Incorporating MNE-Pyblinker
into the MNE-python toolkit—an established platform for EEG and MEG data analysis—would democratize access to efficient eyeblink

artifact detection. This transition is poised to enhance research productivity and integrate seamlessly with Python-based
analytical workflows. Consequently, converting BLINKERS into a Python-friendly tool emerges as a pivotal endeavor to eliminate the
barriers imposed by programming language discrepancies, thereby propelling neuroscience research methodologies towards greater
innovation and efficiency.


# Hypothesis
1. The migration of BLINKERS from MATLAB to a Python-compatible version will achieve equivalent or superior performance due to
   the incorporation of diverse datasets from various hardware settings and populations, thereby enhancing its generalizability.
3. Research Questions
1. How does the migration of BLINKER to a Python-compatible version (MNE-Pyblinker) affect the efficiency of EEG data
   preprocessing in terms of time required for identifying eye-blink artifacts compared to traditional manual methods and the original
   MATLAB version?
2. Does the integration of diverse datasets from other hardware settings and populations into the training process of MNE-Pyblinker
   improve the generalizability of the artifact detection compared to the original MATLAB-version?
3. To what extent does the parameter tuning in MNE-Pyblinker adapt to the variability in EEG data due to different hardware and
   demographic factors, and how does this adaptability affect the overall performance of the system?

# Literature Review
   There is more to eye movement than meets the eye, and this motivates researchers to uncover the trend in electrical brain activity
   associated with eye behavior. As for many physiological phenomena, there is a recently demonstrable indication that the eye-blink of
   a healthy human is influenced by cognitive processes (Alyan et al., 2023a, 2023b). Research suggests that ocular indices
   demonstrate unique signatures during different psychological tasks and cognitive load levels. Additionally, it has been shown that a
   person with Parkinson's disease, Schizophrenia, or under medication exhibit abnormal blink pattern compared to a healthy individual
   (Kimura et al., 2017; Mentzel et al., 2017). Therefore, the ocular index is a promising surrogate parameter in providing insight into a
   person's neurocognitive mechanisms.
   While the presence of eye movement in the EEG raw signal is generally considered an unwanted artifact, this opens the possibility to
   derive different blink characteristics directly from EEG modality alone. In addition, instead of the conventional practice which uses a
   separate video camera to record the eye activity, the nondependent on the video camera solely for blink detection may subsequently
   simplify the overall experiment setup.
   The most common methodological approach in understanding co-registration of brain activity with eye movements is by examining
   the EEG activity time-lock to certain eye movement events (Alyan et al., 2023b; Antipov, 2023). Marking eye blink events can be
   challenging mainly to the fact that it requires manual annotation of the raw EEG signal, and the effort is increased significantly when
   dealing with a large dataset (Kleifges et al., 2017). Additionally, manual annotation is skill-dependent since the eye movement signal
   does not possess consistent statistical and spectral properties, which consequently makes them challenging to segregate from other
   signals such as EEG (Jansen et al., 2021).
   BLINKERS is an algorithm for automatically extracting ocular indices from EEG, electrooculography channels, and/or independent
   components (ICs) (Kleifges et al., 2017). The package is equips with the ability to automatically discover which 
   channels or independent components capture blinks, as well as calculate standard ocular indices, generate a report for each dataset, dump
   labeled images of the individual blinks, and provide summary statistics across collections. However, BLINKER is only available in
   MATLAB, which indirectly requires a MATLAB license to be used and limits its usage among Python neuroscience researchers who
   have seen a dramatic increase in user base in the previous decade. Additionally, BLINKER performance is not consistent when used
   with specific recording recorded from different manufacturers.
   MNE-Python is an open-source software package and a general-purpose package that enable easy exploration, visualization, and
   analysis of human neurophysiological data, including EEG (Gramfort et al., 2013). The package was initially started as a Python port
   of a software package called MNE, which was developed in the C language. Since its first inception, MNE-Python has then capable
   of performing a wider range of processing and visualization tasks. The user of MNE-Python enjoys continuous software improvement
   and usage support primarily due to the support it receives from independent contributors that come from around the globe.
   Additionally, the package development is also supported via funding from institutions such as the National Institutes of Health, the
   Chan Zuckerberg open-source initiative, the European Research Council, and Agence Nationale de la Recherche in France,
   In recent years, there has been strong interest in integrating the neuroscience package into the MNE-Python, which is the de-facto
   neurophysiological data analysis package in python (Appelhoff et al., 2019). Such integration allows researchers to go from raw EEG
   data to visualizations and analysis using an all-in-one package yet strictly obey modern coding best practices. The package that
   tightly integrated with MNE-Python benefited from the immediate access to a wide variety of algorithms both from within the MNEPython

suits and other packages such as MNE-BIDS, MNE-ICALABEL, and MNE-AUTOREJECT. Hence, this project aims to
migrate the MATLAB-version BLINKER to a Python-compatible environment under the code name MNE-Pyblinker. The development
of a simple application programming interface (API) for the MNE-Pyblinker will facilitate easy adaptation to improve the algorithm and
spark an interest in new developers and contributors to maintain and improve this package further.

# Methodology:
1. Description of Methodology
   The methodology of this project is designed to transition and enhance the capabilities of BLINKER features from MATLAB to
   Python, specifically integrating with MNE-Python for EEG data analysis. It is structured into two distinct phases, each targeting a
   specific set of objectives that collectively aim to not only migrate but also improve upon the original functionality.
   Phase I: Migration and Integration
   The first phase of the project is dedicated to migrating BLINKER features, originally developed in MATLAB, to the Python
   programming environment. This involves a detailed process of re-coding the features to be compatible with Python syntax and
   libraries, with a particular focus on ensuring seamless integration with MNE-Python, a widely-used library for processing
   electrophysiological data including EEG. Key steps and strategies in this phase include:
   Phase I A) Component Migration
   Each BLINKER feature will be individually ported to Python, with careful attention to maintaining the integrity and functionality of the
   original MATLAB code.
   Phase I B) Unit Testing
   Following migration, each component will undergo rigorous unit testing. This is a critical step designed to identify and rectify any
   errors or discrepancies in the code at an early stage, ensuring reliability and accuracy in the migrated features.
   Phase I C) Validation through Comparison
   To ensure that the migrated BLINKER features (MNE-Pyblinker) produce equivalent results to the original MATLAB version, a
   thorough comparison of outputs will be conducted. This involves running parallel analyses using both versions on the same datasets
   and verifying that the outcomes match, thereby validating the success of the migration process.
   Phase II: Enhancement and Generalization
   The second phase focuses on improving the functionality of MNE-Pyblinker, particularly in terms of its adaptability and effectiveness
   across different EEG recording specifications. This enhancement phase includes:
   Phase II A) Development of a New Preprocessing Pipeline
   Recognizing the diversity in EEG recording specifications, a new preprocessing pipeline will be introduced to MNE-Pyblinker. This
   pipeline will be designed to efficiently handle data from other EEG acquisition systems, ensuring robust performance across different
   setups.
   Phase II B) Optimization of Algorithm-Free Parameters
   To enhance the generalization capability of MNE-Pyblinker, the project will optimize algorithm-free parameters. To support this
   optimization process, the project aims to publicly available resting-state EEG dataset. This diverse collection of data will provide a
   robust foundation for refining the algorithm, ensuring that MNE-Pyblinker can accurately and effectively process EEG data across a
   broad spectrum of recording conditions. This phase is critical for enhancing the tool's utility and applicability in various research and
   clinical settings, ultimately contributing to the advancement of EEG analysis methodologies.

# Research deliverables
- The introduction of a new preprocessing pipeline could lead to the discovery of innovative techniques for artifact removal that are
  more effective or efficient than existing practice. These techniques could be particularly adept at handling artifacts in data from the
  newly considered EEG acquisition systems, enriching the toolbox available to researchers and clinicians alike.
- In addition, the project's efforts to standardize and enhance EEG signal processing might lead to improvements in the reliability and
  validity of existing biomarkers, making them more robust across different study designs and populations.
# References

1. Alyan, E., Arnau, S., Reiser, J. E., Getzmann, S., Karthaus, M., & Wascher, E. (2023a). Blink-related EEG activity measures cog
   nitive load during proactive and reactive driving. Scientific Reports, 13(1). https://doi.org/10.1038/s41598-023-46738-0
2. Alyan, E., Arnau, S., Reiser, J. E., Getzmann, S., Karthaus, M., & Wascher, E. (2023b). Decoding Eye Blink and Related EEG A
   ctivity in Realistic Working Environments. IEEE Journal of Biomedical and Health Informatics, 27(12), 5745-5754. https://doi.org/10.1
   109/JBHI.2023.3317508
3. Antipov, V. (2023). Detecting fatigue indicators from electroencephalogram data during prolonged cognitive load.
4. Appelhoff, S., Sanderson, M., Brooks, T. L., van Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höch
   enberger, R., Welke, D., Brunner, C., Rockhill, A. P., Larson, E., Gramfort, A., & Jas, M. (2019). MNE-BIDS: Organizing electrophysi
   ological data into the BIDS format and facilitating their analysis. Journal of Open Source Software, 4(44), 1896-1896. https://doi.org/1
   0.21105/joss.01896
5. Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L.,
   Parkkonen, L., & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience(7 DEC). https://
   doi.org/10.3389/fnins.2013.00267
6. Jansen, R. J., van der Kint, S. T., & Hermens, F. (2021). Does agreement mean accuracy? Evaluating glance annotation in natura
   listic driving data. Behavior Research Methods, 53(1), 430-446. https://doi.org/10.3758/s13428-020-01446-9
7. Kimura, N., Watanabe, A., Suzuki, K., Toyoda, H., Hakamata, N., Fukuoka, H., Washimi, Y., Arahata, Y., Takeda, A., Kondo, M.,
   Mizuno, T., & Kinoshita, S. (2017, 2017/09/15/). Measurement of spontaneous blinks in patients with Parkinson's disease using a ne
   w high-speed blink analysis system. Journal of the Neurological Sciences, 380, 200-204. https://doi.org/https://doi.org/10.1016/j.jns.2
   017.07.035
8. Kleifges, K., Bigdely-Shamlo, N., Kerick, S. E., & Robbins, K. A. (2017, 2017-February-03). BLINKER: Automated Extraction of O
   cular Indices from EEG Enabling Large-Scale Analysis [Methods]. Frontiers in Neuroscience, 11. https://doi.org/10.3389/fnins.2017.0
   0012
9. Mentzel, C. L., Bakker, P. R., Van Os, J., Drukker, M., Matroos, G. E., Tijssen, M. A. J., & Van Harten, P. N. (2017). Blink rate is
   associated with drug-induced parkinsonism in patients with severe mental illness, but does not meet requirements to serve as a clini
   cal test: The Curacao extrapyramidal syndromes study XIII. Journal of Negative Results in BioMedicine, 16(1). https://doi.org/10.118
   6/s12952-017-0079-y 