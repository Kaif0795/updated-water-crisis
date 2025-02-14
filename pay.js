document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("paymentForm").addEventListener("submit", function (event) {
        event.preventDefault();

        // Get input values
        const liters = parseFloat(document.getElementById("liters").value);
        const distance = parseFloat(document.getElementById("distance").value);

        // Define rates
        const ratePerLiter = 10; // Rate per liter of water in rupees
        const ratePerKmPerLiter = 0.5; // Rate per kilometer per liter of water

        // Validate input
        if (isNaN(liters) || isNaN(distance) || liters <= 0 || distance < 0) {
            alert("Please enter valid values for liters and distance.");
            return;
        }

        // Calculate the payment
        const waterCost = liters * ratePerLiter;
        const distanceCost = liters * distance * ratePerKmPerLiter;
        const totalPayment = waterCost + distanceCost;

        // Display the result
        document.getElementById("paymentAmount").textContent = `₹${totalPayment.toFixed(2)}`;

        // Show the payment section with animation
        const resultSection = document.getElementById("result");
        resultSection.style.display = "block";
        setTimeout(() => {
            resultSection.classList.add("show");
        }, 50);

        // Show the pay button
        document.getElementById("payButton").style.display = "inline-block";

        // Store total payment to trigger Razorpay on button click
        document.getElementById("payButton").onclick = function () {
            initiatePayment(totalPayment);
        };
    });

    // Function to initiate Razorpay payment with UI-based button
    function initiatePayment(amount) {
        const options = {
            "key": "YOUR_RAZORPAY_KEY", // Your Razorpay Key ID
            "amount": amount * 100, // Amount in paise (1 INR = 100 paise)
            "currency": "INR",
            "name": "Water Payment",
            "description": "Payment for water delivery",
            "image": "https://example.com/logo.png", // Optional logo URL
            "theme": {
                "color": "#4CAF50"
            },
            "handler": function (response) {
                alert("Payment successful! Payment ID: " + response.razorpay_payment_id);
            },
            "prefill": {
                "name": "Customer Name",
                "email": "customer@example.com",
                "contact": "1234567890"
            }
        };

        const rzp1 = new Razorpay(options);
        rzp1.open();
    }
});
