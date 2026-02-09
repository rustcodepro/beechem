mod args;
mod qsar;
use crate::args::CommandParse;
use crate::args::Commands;
use crate::qsar::qsar_chem;
use clap::Parser;
use figlet_rs::FIGfont;

/*
Gaurav Sablok
codeprog@icloud.com
*/

fn main() {
    let fontgenerate = FIGfont::standard().unwrap();
    let repgenerate = fontgenerate.convert("beechem");
    println!("{}", repgenerate.unwrap());

    let args = CommandParse::parse();
    match &args.command {
        Commands::QSARElasticNet {
            filepath,
            nfold,
            thread,
        } => {
            let n_threads = thread.parse::<usize>().expect("thread must be a number");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to create thread pool");
            pool.install(|| {
                let moduleqqsar = qsar_chem(filepath, nfold).unwrap();

                println!("The qsar modelling has finished: {}", moduleqqsar);
            });
        }
    }
}
