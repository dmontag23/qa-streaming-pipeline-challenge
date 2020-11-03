import './App.css';
import axios from "axios";

function runScript() {

  console.log("The script was run!")
  axios.get("/api/v1/data?page=0")
    .then(function (response) {
    // handle success
    console.log(response);
    console.log(response.data)
    })
    .catch(function (error) {
      // handle error
    console.log(error);
    })
    .then(function () {
    // always executed
    });
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
