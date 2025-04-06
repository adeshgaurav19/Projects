document.addEventListener('DOMContentLoaded', function() {
    const issueTenderForm = document.getElementById('issueTenderForm');
    const submitBidForm = document.getElementById('submitBidForm');
    const finalizeContractForm = document.getElementById('finalizeContractForm');
    
    // Issue Tender Form submission
    if (issueTenderForm) {
        issueTenderForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(issueTenderForm);
            const data = {
                title: formData.get('title'),
                description: formData.get('description'),
                budget: formData.get('budget'),
                deadline: formData.get('deadline')
            };
            
            fetch('/issue_tender', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            }).then(response => response.json())
              .then(result => {
                  alert('Tender issued successfully!');
                  window.location.href = '/';
              });
        });
    }

    // Submit Bid Form submission
    if (submitBidForm) {
        submitBidForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(submitBidForm);
            const data = {
                tender_id: formData.get('tender_id'),
                supplier_name: formData.get('supplier_name'),
                bid_amount: formData.get('bid_amount'),
                delivery_time: formData.get('delivery_time')
            };
            
            fetch('/submit_bid', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            }).then(response => response.json())
              .then(result => {
                  alert('Bid submitted successfully!');
                  window.location.href = '/';
              });
        });
    }

    // Finalize Contract Form submission
    if (finalizeContractForm) {
        finalizeContractForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(finalizeContractForm);
            const data = {
                tender_id: formData.get('tender_id'),
                supplier_name: formData.get('supplier_name')
            };
            
            fetch('/finalize_contract', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            }).then(response => response.json())
              .then(result => {
                  alert('Contract finalized!');
                  window.location.href = '/';
              });
        });
    }
});
