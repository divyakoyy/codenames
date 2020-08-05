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

      const selectedStyle = {
        background: '#4183cc',
        color: 'white',
        height: 100,
        width: '100%',
        fontSize: 20,
        fontWeight: 'bold',
        border: 'none',
        textTransform: 'uppercase',
      };

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

var words = [
        'vacuum', 'whip', 'moon', 'school', 'tube', 'lab', 'key', 'table', 'lead', 'crown',
        'bomb', 'bug', 'pipe', 'roulette','australia', 'play', 'cloak', 'piano', 'beijing', 'bison',
        'boot', 'cap', 'car','change', 'circle', 'cliff', 'conductor', 'cricket', 'death', 'diamond',
        'figure', 'gas', 'germany', 'india', 'jupiter', 'kid', 'king', 'lemon', 'litter', 'nut',
        'phoenix', 'racket', 'row', 'scientist', 'shark', 'stream', 'swing', 'unicorn', 'witch', 'worm',
        'pistol', 'saturn', 'rock', 'superhero', 'mug', 'fighter', 'embassy', 'cell', 'state', 'beach',
        'capital', 'post', 'cast', 'soul', 'tower', 'green', 'plot', 'string', 'kangaroo', 'lawyer',
        ];

shuffle(words);

// Find all DOM containers, and render Like buttons into them.
document.querySelectorAll('.codeword-button')
  .forEach(domContainer => {
    // Read the comment ID from a data-* attribute.
    const commentID = parseInt(domContainer.dataset.commentid, 10);
    ReactDOM.render(
      e(CodewordButton, { commentID: commentID, wordLabel: words[commentID] }),
      domContainer
    );
  });

