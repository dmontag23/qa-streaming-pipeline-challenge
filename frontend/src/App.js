import './App.css';
import axios from "axios";

function runScript() {

  console.log("The script was run!")
  axios.get("/api/v1/run_script")
    .then(function (response) {
    // handle success
    console.log(response);
    console.log(response.data[0].classification)
    console.log(response.data[0].regression)
    })
    .catch(function (error) {
      // handle error
    console.log(error);
    })
}

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <button onClick={runScript}>
            Run Script
        </button>
      </header>
    </div>
  );
}

export default App;
