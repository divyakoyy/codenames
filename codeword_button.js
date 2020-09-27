'use strict';

const e = React.createElement;

class CodewordButton extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      selected: false 
    };
  }

  render() {

    if (this.state.selected) {

      var isBlueWord = blue_words.includes(this.props.wordLabel);
      var selectedStyle = {}
      if (isBlueWord) {
        selectedStyle = {
          background: '#4183cc',
          color: 'white',
          height: 100,
          width: '100%',
          fontSize: 20,
          fontWeight: 'bold',
          border: 'none',
          textTransform: 'uppercase',
        };
      } else {
        selectedStyle = {
          background: '#d13030',
          color: 'white',
          height: 100,
          width: '100%',
          fontSize: 20,
          fontWeight: 'bold',
          border: 'none',
          textTransform: 'uppercase',
        };
      }
      
      return e(
        'button',
        { style: selectedStyle },
        this.props.wordLabel
      );
    }

    const divStyle = {
      height: 100,
      width: '100%',
      fontSize: 20,
      border: 'none',
      textTransform: 'uppercase',
    };


    return e(
      'button',
      { onClick: () => this.setState({ selected: true }), style: divStyle },
      this.props.wordLabel
    );
  }
}

/**
 * Shuffles array in place. ES6 version
 * @param {Array} a items An array containing the items.
 */
function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

var blue_words = ['bill','witch','himalayas', 'straw', 'greece',  'circle', 'pyramid', 'mug', 'scale', 'contract'];
var red_words = [ 'spot',  'washer',  'tap', 'ray', 'bison', 'brush', 'nurse', 'compound', 'lock',  'doctor'];
var words = blue_words.concat(red_words);
shuffle(words);

// Find all DOM containers, and render Like buttons into them.
document.querySelectorAll('.codeword-button')
  .forEach(domContainer => {
    // Read the comment ID from a data-* attribute.
    const commentID = parseInt(domContainer.dataset.commentid, 10);
    ReactDOM.render(
      e(CodewordButton, { commentID: commentID, wordLabel: words[commentID-1] }),
      domContainer
    );
  });

