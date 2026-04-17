"""Extract intermediate CSV tables from raw eICU CSVs.

This is the first stage of the TPC preprocessing pipeline. It replaces
the PostgreSQL SQL stage (create_all_tables.sql) by reading raw eICU CSVs
directly and producing the same intermediate CSV files (labels, timeseries,
flat features, diagnoses) needed by subsequent preprocessing steps.

Usage:
    python tpc/extract_tables.py --eicu-dir /path/to/eICU/csvs --output-dir /path/to/output
"""
import argparse
import pandas as pd
import numpy as np
import os


# Loose cross-cohort prevalence floor applied at Stage A (extraction). The
# tight double-threshold prevalence filter runs in apply_filters.py so it can
# be re-tuned without re-extracting. 5% is safely below any reasonable
# Stage-B double-threshold setting — see design notes.
STAGE_A_PREFILTER = 0.05


def load_csv(eicu_dir, name):
    path = os.path.join(eicu_dir, name)
    if not os.path.exists(path):
        gz_path = path + '.gz'
        if os.path.exists(gz_path):
            path = gz_path
    print(f'  Loading {os.path.basename(path)}...')
    return pd.read_csv(path, low_memory=False)


def create_labels(patient, apr):
    """Build the labels/cohort table.

    Cohort rules:
      - adults only (age > 17; '> 89' kept as 89)
      - ICU stay longer than 5 hours (actualiculos computed from
        unitdischargeoffset so patients without an APACHE record are kept)

    APACHE predicted fields are left-joined (IVa preferred) and may be NaN.
    """
    print('==> Creating labels...')

    labels = patient.copy()

    # Actual ICU LoS in days, derived from unitdischargeoffset (minutes).
    labels['actualiculos'] = labels['unitdischargeoffset'] / 1440.0

    # Age filter with '> 89' handling.
    age_numeric = labels['age'].replace('> 89', '89')
    age_numeric = pd.to_numeric(age_numeric, errors='coerce')
    labels = labels[age_numeric > 17]

    # >5h ICU length-of-stay floor.
    labels = labels[labels['actualiculos'] > (5 / 24)]

    # Left-join APACHE fields (prefer IVa). Patients without an APACHE record
    # are retained with NaN in the predicted columns.
    apr_iva = apr[apr['apacheversion'] == 'IVa'][
        ['patientunitstayid', 'predictedhospitalmortality', 'predictediculos',
         'actualhospitalmortality']
    ].drop_duplicates(subset='patientunitstayid', keep='first')
    labels = labels.merge(apr_iva, on='patientunitstayid', how='left')

    # Fall back to patient.hospitaldischargestatus for the actual-mortality
    # column when APACHE didn't record one. Values in patient are 'Expired' /
    # 'Alive' → uppercased to match APACHE's 'EXPIRED' / 'ALIVE' convention.
    missing = labels['actualhospitalmortality'].isnull()
    if missing.any():
        fallback = labels.loc[missing, 'hospitaldischargestatus'].str.upper()
        labels.loc[missing, 'actualhospitalmortality'] = fallback

    # Drop stays where neither source provided an outcome — the mortality task
    # can't use them and a NaN label would break downstream label dtype.
    before = len(labels)
    labels = labels[labels['actualhospitalmortality'].isin(['ALIVE', 'EXPIRED'])]
    if before != len(labels):
        print(f'  Dropped {before - len(labels)} stays with unknown mortality outcome')

    labels = labels[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid',
                      'unitvisitnumber', 'unitdischargelocation', 'unitdischargeoffset',
                      'unitdischargestatus', 'predictedhospitalmortality',
                      'actualhospitalmortality', 'predictediculos', 'actualiculos']].copy()

    print(f'  Labels: {len(labels)} rows (APACHE IVa filter removed)')
    return labels


def create_timeseries_lab(patient, lab, labels):
    """Replicate ld_commonlabs and ld_timeserieslab views."""
    print('==> Creating timeseries lab...')

    label_pids = set(labels['patientunitstayid'])
    discharge_offsets = patient[['patientunitstayid', 'unitdischargeoffset']].drop_duplicates()

    # Join lab with patient to get discharge offset, filter to cohort
    lab_filtered = lab[lab['patientunitstayid'].isin(label_pids)].copy()
    lab_filtered = lab_filtered.merge(discharge_offsets, on='patientunitstayid', how='inner')

    # Filter offset between -1440 and unitdischargeoffset
    lab_filtered = lab_filtered[
        (lab_filtered['labresultoffset'] >= -1440) &
        (lab_filtered['labresultoffset'] <= lab_filtered['unitdischargeoffset'])
    ]

    # Loose global pre-filter (5%). The tight double-threshold prevalence
    # filter runs later in apply_filters.py so researchers can re-tune it
    # without re-extracting and re-binning.
    num_patients = labels['patientunitstayid'].nunique()
    lab_counts = lab_filtered.groupby('labname')['patientunitstayid'].nunique()
    common_labs = lab_counts[lab_counts > num_patients * STAGE_A_PREFILTER].index
    print(f'  Labs passing {STAGE_A_PREFILTER*100:.0f}% pre-filter: {len(common_labs)}')

    # Filter to common labs only
    ts_lab = lab_filtered[lab_filtered['labname'].isin(common_labs)][
        ['patientunitstayid', 'labresultoffset', 'labname', 'labresult']
    ].copy()

    print(f'  Timeseries lab: {len(ts_lab)} rows')
    return ts_lab


def create_timeseries_resp(patient, resp, labels):
    """Replicate ld_commonresp and ld_timeseriesresp views."""
    print('==> Creating timeseries respiratory...')

    label_pids = set(labels['patientunitstayid'])
    discharge_offsets = patient[['patientunitstayid', 'unitdischargeoffset']].drop_duplicates()

    resp_filtered = resp[resp['patientunitstayid'].isin(label_pids)].copy()
    resp_filtered = resp_filtered.merge(discharge_offsets, on='patientunitstayid', how='inner')
    resp_filtered = resp_filtered[
        (resp_filtered['respchartoffset'] >= -1440) &
        (resp_filtered['respchartoffset'] <= resp_filtered['unitdischargeoffset'])
    ]

    num_patients = labels['patientunitstayid'].nunique()
    resp_counts = resp_filtered.groupby('respchartvaluelabel')['patientunitstayid'].nunique()
    common_resp = resp_counts[resp_counts > num_patients * STAGE_A_PREFILTER].index
    print(f'  Resp features passing {STAGE_A_PREFILTER*100:.0f}% pre-filter: {len(common_resp)}')

    ts_resp = resp_filtered[resp_filtered['respchartvaluelabel'].isin(common_resp)][
        ['patientunitstayid', 'respchartoffset', 'respchartvaluelabel', 'respchartvalue']
    ].copy()

    print(f'  Timeseries resp: {len(ts_resp)} rows')
    return ts_resp


def create_timeseries_nurse(patient, nurse, labels):
    """Replicate ld_commonnurse and ld_timeseriesnurse views."""
    print('==> Creating timeseries nurse...')

    label_pids = set(labels['patientunitstayid'])
    discharge_offsets = patient[['patientunitstayid', 'unitdischargeoffset']].drop_duplicates()

    # nurseCharting is very large, filter early
    nurse_filtered = nurse[nurse['patientunitstayid'].isin(label_pids)].copy()
    del nurse
    nurse_filtered = nurse_filtered.merge(discharge_offsets, on='patientunitstayid', how='inner')
    nurse_filtered = nurse_filtered[
        (nurse_filtered['nursingchartoffset'] >= -1440) &
        (nurse_filtered['nursingchartoffset'] <= nurse_filtered['unitdischargeoffset'])
    ]

    num_patients = labels['patientunitstayid'].nunique()
    nurse_counts = nurse_filtered.groupby('nursingchartcelltypevallabel')['patientunitstayid'].nunique()
    common_nurse = nurse_counts[nurse_counts > num_patients * STAGE_A_PREFILTER].index
    print(f'  Nurse features passing {STAGE_A_PREFILTER*100:.0f}% pre-filter: {len(common_nurse)}')

    ts_nurse = nurse_filtered[nurse_filtered['nursingchartcelltypevallabel'].isin(common_nurse)][
        ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel', 'nursingchartvalue']
    ].copy()

    print(f'  Timeseries nurse: {len(ts_nurse)} rows')
    return ts_nurse


def create_timeseries_periodic(patient, vp, labels):
    """Replicate ld_timeseriesperiodic view."""
    print('==> Creating timeseries periodic...')

    label_pids = set(labels['patientunitstayid'])
    discharge_offsets = patient[['patientunitstayid', 'unitdischargeoffset']].drop_duplicates()

    vp_filtered = vp[vp['patientunitstayid'].isin(label_pids)].copy()
    del vp
    vp_filtered = vp_filtered.merge(discharge_offsets, on='patientunitstayid', how='inner')
    vp_filtered = vp_filtered[
        (vp_filtered['observationoffset'] >= -1440) &
        (vp_filtered['observationoffset'] <= vp_filtered['unitdischargeoffset'])
    ]

    cols = ['patientunitstayid', 'observationoffset', 'temperature', 'sao2', 'heartrate',
            'respiration', 'cvp', 'systemicsystolic', 'systemicdiastolic', 'systemicmean',
            'st1', 'st2', 'st3']
    ts_periodic = vp_filtered[cols].sort_values(['patientunitstayid', 'observationoffset']).copy()

    print(f'  Timeseries periodic: {len(ts_periodic)} rows')
    return ts_periodic


def create_timeseries_aperiodic(patient, va, labels):
    """Replicate ld_timeseriesaperiodic view."""
    print('==> Creating timeseries aperiodic...')

    label_pids = set(labels['patientunitstayid'])
    discharge_offsets = patient[['patientunitstayid', 'unitdischargeoffset']].drop_duplicates()

    va_filtered = va[va['patientunitstayid'].isin(label_pids)].copy()
    va_filtered = va_filtered.merge(discharge_offsets, on='patientunitstayid', how='inner')
    va_filtered = va_filtered[
        (va_filtered['observationoffset'] >= -1440) &
        (va_filtered['observationoffset'] <= va_filtered['unitdischargeoffset'])
    ]

    cols = ['patientunitstayid', 'observationoffset', 'noninvasivesystolic',
            'noninvasivediastolic', 'noninvasivemean']
    ts_aperiodic = va_filtered[cols].sort_values(['patientunitstayid', 'observationoffset']).copy()

    print(f'  Timeseries aperiodic: {len(ts_aperiodic)} rows')
    return ts_aperiodic


def create_flat_features(patient, aps, apr, apv, hospital, labels):
    """Replicate ld_flat view."""
    print('==> Creating flat features...')

    label_pids = set(labels['patientunitstayid'])

    # Start with patient table, filter to cohort
    p = patient[patient['patientunitstayid'].isin(label_pids)].copy()

    # Extract hour from unitadmittime24
    p['hour'] = p['unitadmittime24'].apply(
        lambda x: int(str(x).split(':')[0]) if pd.notnull(x) else np.nan
    )

    # Join all tables
    flat = p.merge(aps[['patientunitstayid', 'intubated', 'vent', 'dialysis',
                         'eyes', 'motor', 'verbal', 'meds']],
                    on='patientunitstayid', how='inner')
    flat = flat.merge(apr[['patientunitstayid', 'physicianspeciality']].drop_duplicates(),
                       on='patientunitstayid', how='inner')
    flat = flat.merge(apv[['patientunitstayid', 'bedcount']],
                       on='patientunitstayid', how='inner')
    flat = flat.merge(hospital[['hospitalid', 'numbedscategory', 'teachingstatus', 'region']],
                       on='hospitalid', how='inner')

    # Select and deduplicate
    cols = ['patientunitstayid', 'gender', 'age', 'ethnicity', 'admissionheight',
            'admissionweight', 'apacheadmissiondx', 'hour', 'unittype', 'unitadmitsource',
            'unitvisitnumber', 'unitstaytype', 'physicianspeciality', 'intubated', 'vent',
            'dialysis', 'eyes', 'motor', 'verbal', 'meds', 'bedcount', 'numbedscategory',
            'teachingstatus', 'region']
    flat = flat[cols].drop_duplicates()

    print(f'  Flat features: {len(flat)} rows')
    return flat


def create_diagnoses(patient, diagnosis, pasthistory, admissiondx, labels):
    """Replicate ld_diagnoses view."""
    print('==> Creating diagnoses...')

    label_pids = set(labels['patientunitstayid'])

    # Current diagnoses within first 5 hours
    diag = diagnosis[diagnosis['patientunitstayid'].isin(label_pids)].copy()
    diag = diag[diag['diagnosisoffset'] < 60 * 5]
    diag_out = diag[['patientunitstayid', 'diagnosisstring']].copy()

    # Past history within first 5 hours
    ph = pasthistory[pasthistory['patientunitstayid'].isin(label_pids)].copy()
    ph = ph[ph['pasthistoryoffset'] < 60 * 5]
    ph_out = ph[['patientunitstayid']].copy()
    ph_out['diagnosisstring'] = ph['pasthistorypath']

    # Admission diagnoses within first 5 hours
    ad = admissiondx[admissiondx['patientunitstayid'].isin(label_pids)].copy()
    ad = ad[ad['admitdxenteredoffset'] < 60 * 5]
    ad_out = ad[['patientunitstayid']].copy()
    ad_out['diagnosisstring'] = ad['admitdxpath']

    # Union (like SQL UNION = distinct)
    all_diag = pd.concat([diag_out, ph_out, ad_out], ignore_index=True).drop_duplicates()

    print(f'  Diagnoses: {len(all_diag)} rows')
    return all_diag


def create_timeseries_patients(ts_lab, ts_resp, ts_nurse, ts_periodic, ts_aperiodic):
    """Replicate ld_timeseries_patients view - union of all patients with any timeseries."""
    print('==> Finding patients with timeseries data...')
    pids = set()
    for df in [ts_lab, ts_resp, ts_nurse, ts_periodic, ts_aperiodic]:
        pids.update(df['patientunitstayid'].unique())
    print(f'  Patients with timeseries: {len(pids)}')
    return pids


EXPECTED_OUTPUTS = (
    'labels.csv', 'diagnoses.csv', 'flat_features.csv',
    'timeserieslab.csv', 'timeseriesresp.csv', 'timeseriesnurse.csv',
    'timeseriesperiodic.csv', 'timeseriesaperiodic.csv',
)


def main(eicu_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Cache check: if every Stage 1 output is already present and non-empty,
    # skip re-extraction. Pass --force to preprocessing.py (which removes the
    # whole output dir) to bypass.
    all_present = all(
        os.path.exists(os.path.join(output_dir, f))
        and os.path.getsize(os.path.join(output_dir, f)) > 0
        for f in EXPECTED_OUTPUTS
    )
    if all_present:
        print(f'[cached] Stage 1 outputs already present in {output_dir}, skipping extraction.')
        return

    print('=' * 60)
    print('Extract tables: eICU preprocessing (Stage 1)')
    print(f'Raw data: {eicu_dir}')
    print(f'Output:   {output_dir}')
    print('=' * 60)

    # Load raw tables
    print('\n==> Loading raw eICU tables...')
    patient = load_csv(eicu_dir, 'patient.csv')
    apr = load_csv(eicu_dir, 'apachePatientResult.csv')

    # Step 1: Labels (must be first)
    labels = create_labels(patient, apr)

    # Step 2: Timeseries (load heavy tables one at a time to manage memory)
    lab = load_csv(eicu_dir, 'lab.csv')
    ts_lab = create_timeseries_lab(patient, lab, labels)
    del lab

    resp = load_csv(eicu_dir, 'respiratoryCharting.csv')
    ts_resp = create_timeseries_resp(patient, resp, labels)
    del resp

    nurse = load_csv(eicu_dir, 'nurseCharting.csv')
    ts_nurse = create_timeseries_nurse(patient, nurse, labels)
    # nurse is deleted inside the function

    vp = load_csv(eicu_dir, 'vitalPeriodic.csv')
    ts_periodic = create_timeseries_periodic(patient, vp, labels)
    # vp is deleted inside the function

    va = load_csv(eicu_dir, 'vitalAperiodic.csv')
    ts_aperiodic = create_timeseries_aperiodic(patient, va, labels)

    # Step 3: Flat features
    aps = load_csv(eicu_dir, 'apacheApsVar.csv')
    apv = load_csv(eicu_dir, 'apachePredVar.csv')
    hospital = load_csv(eicu_dir, 'hospital.csv')
    flat = create_flat_features(patient, aps, apr, apv, hospital, labels)
    del aps, apv, hospital

    # Step 4: Diagnoses
    diagnosis = load_csv(eicu_dir, 'diagnosis.csv')
    pasthistory = load_csv(eicu_dir, 'pastHistory.csv')
    admissiondx = load_csv(eicu_dir, 'admissionDx.csv')
    diagnoses = create_diagnoses(patient, diagnosis, pasthistory, admissiondx, labels)
    del diagnosis, pasthistory, admissiondx

    # Step 5: Filter to patients who have timeseries data
    ts_patients = create_timeseries_patients(ts_lab, ts_resp, ts_nurse, ts_periodic, ts_aperiodic)
    labels_out = labels[labels['patientunitstayid'].isin(ts_patients)]
    flat_out = flat[flat['patientunitstayid'].isin(ts_patients)]
    diagnoses_out = diagnoses[diagnoses['patientunitstayid'].isin(ts_patients)]

    # Step 6: Save all CSVs
    print('\n==> Saving output CSVs...')

    def save(df, name):
        path = os.path.join(output_dir, name)
        df.to_csv(path, index=False)
        print(f'  Saved {name} ({len(df)} rows)')

    save(labels_out, 'labels.csv')
    save(diagnoses_out, 'diagnoses.csv')
    save(flat_out, 'flat_features.csv')
    save(ts_lab, 'timeserieslab.csv')
    save(ts_resp, 'timeseriesresp.csv')
    save(ts_nurse, 'timeseriesnurse.csv')
    save(ts_periodic, 'timeseriesperiodic.csv')
    save(ts_aperiodic, 'timeseriesaperiodic.csv')

    print('\n==> Stage 1 complete!')
    print(f'Output files written to: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract intermediate CSV tables from raw eICU CSVs.')
    parser.add_argument('--eicu-dir', required=True,
                        help='Path to directory containing raw eICU CSV files.')
    parser.add_argument('--output-dir', required=True,
                        help='Path to directory where intermediate CSVs will be written.')
    args = parser.parse_args()
    main(args.eicu_dir, args.output_dir)
