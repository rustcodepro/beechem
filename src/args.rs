use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "Beechem",
    version = "1.0",
    about = "QSAR modelling of the chemical compounds
       ************************************************
       Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// estimation sheet
    QSARElasticNet {
        /// path to the file
        filepath: String,
        /// nfold value
        nfold: String,
        /// threads for the analysis
        thread: String,
    },
}
