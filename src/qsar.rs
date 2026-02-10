use linfa::dataset::Dataset;
use linfa::prelude::*;
use linfa::traits::Fit;
use linfa_elasticnet::ElasticNet;
use ndarray::Array1;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
codeprog@icloud.com
*/

/*
let descriptor_cols = ["MW", "LogP", "TPSA", "HBD", "HBA"];
let target_col = "pIC50";
let (x, y) = read_csv_descriptors(pathfile, &descriptor_cols, target_col).unwrap();
let (a, b) = to_ndarrays(x, y).unwrap();
let dataset = Dataset::new(a, b);

*/

pub fn qsar_chem(path: &str, nfoldvalue: &str) -> Result<String, Box<dyn Error>> {
    let pathfile = path;
    let penalties = vec![0.001, 0.01, 0.1, 1.0];
    let l1_ratios = vec![0.0, 0.3, 0.5, 0.7, 1.0]; // 0 = ridge, 1 = lasso
    let n_folds = nfoldvalue.parse::<usize>().unwrap();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_params: Option<(f64, f64)> = None;
    let mut best_model: Option<ElasticNet<f64>> = None;
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut matrixvec: Vec<Vec<f64>> = Vec::new();
    let mut matrixy: Vec<f64> = Vec::new();
    for i in fileread.lines() {
        let line = i.expect("fline not present");
        let linevec = line.split("\t").collect::<Vec<_>>();
        let matrixvector = linevec[0..linevec.len() - 1]
            .to_vec()
            .iter()
            .map(|x| x.parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        let matrixlastcall = linevec[linevec.len() - 1..linevec.len()].to_vec()[0]
            .parse::<f64>()
            .unwrap();
        matrixvec.push(matrixvector);
        matrixy.push(matrixlastcall);
    }
    let xrows = matrixvec.len();
    let xcols = matrixvec[0].len();
    let xflat: Vec<f64> = matrixvec.into_iter().flatten().collect();
    let xarray = ndarray::Array2::from_shape_vec((xrows, xcols), xflat).unwrap();
    let yarray = ndarray::Array1::from_vec(matrixy);
    let mut finaldataset = Dataset::new(xarray, yarray);
    let samplecount = finaldataset.nsamples();

    // all hyperparameter combinations
    for &penalty in &penalties {
        for &l1_ratio in &l1_ratios {
            println!("Testing penalty={:.4}, l1_ratio={:.2}", penalty, l1_ratio);
            let mut cv_r2_scores = Vec::new();
            for (fold_model, valid) in finaldataset.iter_fold(n_folds, |train| {
                ElasticNet::params()
                    .penalty(penalty)
                    .l1_ratio(l1_ratio)
                    .fit(&train)
                    .unwrap()
            }) {
                let pred = fold_model.predict(&valid.records);
                let r2 = valid.targets.r2(&pred).unwrap_or(f64::NAN);
                cv_r2_scores.push(r2);
            }
            // average CV
            let mean_r2 = Array1::from_vec(cv_r2_scores).mean().unwrap_or(f64::NAN);
            println!("  → Mean CV R²: {:.4}", mean_r2);
            // Track best
            if mean_r2 > best_score {
                best_score = mean_r2;
                best_params = Some((penalty, l1_ratio));
                // Refit on full dataset with best params (final model)
                best_model = Some(
                    ElasticNet::params()
                        .penalty(penalty)
                        .l1_ratio(l1_ratio)
                        .fit(&finaldataset)?,
                );
            }
        }
    }
    if let (Some((penalty, l1_ratio)), Some(model)) = (best_params, best_model) {
        println!("\n=== Best Parameters (via {n_folds}-fold CV) ===");
        println!("Penalty (λ₂):    {:.4}", penalty);
        println!("L1 ratio:        {:.2}", l1_ratio);
        println!("Best mean CV R²: {:.4}", best_score);
        println!("Confidence: {:?}", model.confidence_95th());
    } else {
        println!("No valid model found.");
    }

    Ok("The qsar model has finished".to_string())
}
